from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile
from scipy.io import loadmat


DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REQUIRED_COLUMNS = {"battery_id", "cycle", "capacity", "temperature"}


def _ensure_processed_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _normalise_history(df: pd.DataFrame, battery_id: str | None = None) -> pd.DataFrame:
    if battery_id is not None and "battery_id" not in df.columns:
        df = df.copy()
        df["battery_id"] = battery_id

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    history = df.loc[:, ["battery_id", "cycle", "capacity", "temperature"]].copy()
    history["battery_id"] = history["battery_id"].astype(str)
    history["cycle"] = pd.to_numeric(history["cycle"], errors="coerce")
    history["capacity"] = pd.to_numeric(history["capacity"], errors="coerce")
    history["temperature"] = pd.to_numeric(history["temperature"], errors="coerce")
    history = history.dropna(subset=["battery_id", "cycle", "capacity", "temperature"])

    if history.empty:
        raise ValueError("No valid battery history rows found")

    history = history.sort_values(["battery_id", "cycle"]).reset_index(drop=True)
    history["soh"] = history.groupby("battery_id")["capacity"].transform(
        lambda values: values / values.iloc[0] * 100
    )
    return history


def _history_files() -> list[Path]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted([*PROCESSED_DIR.glob("*.csv"), *PROCESSED_DIR.glob("*.json")])


def load_all_histories() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in _history_files():
        try:
            if path.suffix == ".csv":
                raw = pd.read_csv(path)
            else:
                raw = pd.read_json(path)
            frames.append(_normalise_history(raw, battery_id=path.stem))
        except Exception:
            continue

    if not frames:
        raise HTTPException(
            status_code=404,
            detail=(
                "No processed NASA battery dataset found. Add processed CSV/JSON files "
                "under backend/data/processed or upload a NASA .mat file."
            ),
        )

    return pd.concat(frames, ignore_index=True)


def available_batteries() -> list[str]:
    try:
        histories = load_all_histories()
    except HTTPException:
        return []
    return sorted(histories["battery_id"].unique().tolist())


def get_history(battery_id: str) -> pd.DataFrame:
    histories = load_all_histories()
    history = histories[histories["battery_id"] == battery_id].copy()

    if history.empty:
        raise HTTPException(status_code=404, detail=f"Battery '{battery_id}' was not found")

    return history.sort_values("cycle").reset_index(drop=True)


def history_response(battery_id: str) -> dict:
    history = get_history(battery_id)
    return {
        "battery_id": battery_id,
        "points": [
            {
                "cycle": int(row.cycle),
                "SOH": round(float(row.soh), 4),
                "capacity": round(float(row.capacity), 6),
                "temperature": round(float(row.temperature), 4),
            }
            for row in history.itertuples()
        ],
    }


def forecast_response(battery_id: str, horizon: int = 40) -> dict:
    history = get_history(battery_id)

    if len(history) < 3:
        raise HTTPException(
            status_code=422,
            detail="At least three historical cycles are required for polynomial forecasting",
        )

    cycles = history["cycle"].to_numpy(dtype=float)
    soh = history["soh"].to_numpy(dtype=float)
    degree = 2 if len(history) >= 3 else 1
    coefficients = np.polyfit(cycles, soh, degree)
    model = np.poly1d(coefficients)

    cycle_step = int(max(1, round(np.median(np.diff(np.sort(cycles))))))
    last_cycle = int(cycles.max())
    future_cycles = np.arange(last_cycle + cycle_step, last_cycle + cycle_step * (horizon + 1), cycle_step)
    predicted_soh = np.clip(model(future_cycles), 0, 120)

    return {
        "battery_id": battery_id,
        "method": "Polynomial degradation baseline",
        "label": "Predicted/estimated future SOH degradation",
        "historical": [
            {"cycle": int(row.cycle), "SOH": round(float(row.soh), 4)}
            for row in history.itertuples()
        ],
        "forecast": [
            {"cycle": int(cycle), "predicted_SOH": round(float(value), 4)}
            for cycle, value in zip(future_cycles, predicted_soh)
        ],
    }


def _as_array(value) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def _extract_field(obj, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def parse_nasa_mat(path: Path, battery_id: str | None = None) -> pd.DataFrame:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [key for key in mat.keys() if not key.startswith("__")]
    if not keys:
        raise ValueError("No MATLAB battery structure found")

    selected_id = battery_id or keys[0]
    battery = mat.get(selected_id, mat[keys[0]])
    cycles = _extract_field(battery, "cycle")

    if cycles is None:
        raise ValueError("MAT file does not contain a NASA battery cycle structure")

    rows = []
    discharge_index = 0

    for cycle in _as_array(cycles):
        cycle_type = str(_extract_field(cycle, "type") or "").lower()
        if cycle_type != "discharge":
            continue

        data = _extract_field(cycle, "data")
        capacity_values = _as_array(_extract_field(data, "Capacity"))
        temperature_values = _as_array(_extract_field(data, "Temperature_measured"))

        if capacity_values.size == 0 or temperature_values.size == 0:
            continue

        discharge_index += 1
        rows.append(
            {
                "battery_id": selected_id,
                "cycle": discharge_index,
                "capacity": float(capacity_values[0]),
                "temperature": float(np.nanmean(temperature_values.astype(float))),
            }
        )

    if not rows:
        raise ValueError("No valid discharge cycles found in MAT file")

    return _normalise_history(pd.DataFrame(rows), battery_id=selected_id)


async def save_uploaded_battery(file: UploadFile) -> dict:
    if not file.filename or not file.filename.lower().endswith(".mat"):
        raise HTTPException(status_code=400, detail="Only NASA .mat files are supported")

    _ensure_processed_dir()
    battery_id = Path(file.filename).stem

    with NamedTemporaryFile(suffix=".mat", delete=True) as temp_file:
        temp_file.write(await file.read())
        temp_file.flush()
        try:
            history = parse_nasa_mat(Path(temp_file.name), battery_id=battery_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    output_path = PROCESSED_DIR / f"{battery_id}.csv"
    history.loc[:, ["battery_id", "cycle", "capacity", "temperature"]].to_csv(output_path, index=False)

    return {
        "battery_id": battery_id,
        "cycles": len(history),
        "history": history_response(battery_id)["points"],
    }
