import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

type BatteryForm = {
  avg_voltage: number;
  min_voltage: number;
  max_voltage: number;
  voltage_range: number;
  voltage_std: number;
  avg_current: number;
  current_std: number;
  avg_temperature: number;
  max_temperature: number;
  temperature_rise: number;
  discharge_time: number;
  discharge_energy_Wh: number;
};

type PredictionResult = {
  SOH: number | string;
  RUL_cycles: number | string;
  anomaly_status: string;
};

type HistoryPoint = {
  cycle: number;
  SOH: number;
  capacity: number;
  temperature: number;
};

type ForecastPoint = {
  cycle: number;
  predicted_SOH: number;
};

type ShapFeature = {
  feature: string;
  value: number;
  shap_value: number;
  impact: string;
};

function App() {
  const [navOpen, setNavOpen] = useState(false);
  const [form, setForm] = useState<BatteryForm>({
    avg_voltage: 3.5,
    min_voltage: 2.5,
    max_voltage: 4.2,
    voltage_range: 1.7,
    voltage_std: 0.3,
    avg_current: -1.9,
    current_std: 0.2,
    avg_temperature: 32,
    max_temperature: 38,
    temperature_rise: 6,
    discharge_time: 3400,
    discharge_energy_Wh: 6.5,
  });

  const [result, setResult] = useState<PredictionResult>({
    SOH: "--",
    RUL_cycles: "--",
    anomaly_status: "--",
  });

  const [loading, setLoading] = useState(false);
  const [batteryIds, setBatteryIds] = useState<string[]>([]);
  const [selectedBattery, setSelectedBattery] = useState("");
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [analyticsMessage, setAnalyticsMessage] = useState("");
  const [shapFeatures, setShapFeatures] = useState<ShapFeature[]>([]);
  const [explanation, setExplanation] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");

  const fields = [
    { key: "avg_voltage", label: "Average voltage", unit: "V" },
    { key: "min_voltage", label: "Minimum voltage", unit: "V" },
    { key: "max_voltage", label: "Maximum voltage", unit: "V" },
    { key: "voltage_range", label: "Voltage range", unit: "V" },
    { key: "voltage_std", label: "Voltage std. dev.", unit: "V" },
    { key: "avg_current", label: "Average current", unit: "A" },
    { key: "current_std", label: "Current std. dev.", unit: "A" },
    { key: "avg_temperature", label: "Average temperature", unit: "C" },
    { key: "max_temperature", label: "Maximum temperature", unit: "C" },
    { key: "temperature_rise", label: "Temperature rise", unit: "C" },
    { key: "discharge_time", label: "Discharge time", unit: "s" },
    { key: "discharge_energy_Wh", label: "Discharge energy", unit: "Wh" },
  ] as const;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;

    setForm((prev) => ({
      ...prev,
      [name]: Number(value),
    }));
  };

  const loadBatteries = async (preferredBattery?: string) => {
    try {
      const response = await fetch("/batteries");
      const data = await response.json();
      const batteries = data.batteries ?? [];

      setBatteryIds(batteries);

      if (batteries.length > 0) {
        setSelectedBattery(
          preferredBattery && batteries.includes(preferredBattery)
            ? preferredBattery
            : batteries[0],
        );
        setAnalyticsMessage("");
      } else {
        setSelectedBattery("");
        setHistory([]);
        setForecast([]);
        setAnalyticsMessage(
          "No NASA battery history is available yet. Upload a NASA .mat file or add processed dataset files on the backend to enable charts.",
        );
      }
    } catch (error) {
      console.error(error);
      setAnalyticsMessage("Could not load battery dataset metadata.");
    }
  };

  useEffect(() => {
    loadBatteries();
  }, []);

  useEffect(() => {
    if (!selectedBattery) {
      return;
    }

    const loadBatteryAnalytics = async () => {
      try {
        const [historyResponse, forecastResponse] = await Promise.all([
          fetch(`/battery-history/${selectedBattery}`),
          fetch(`/battery-forecast/${selectedBattery}`),
        ]);

        if (!historyResponse.ok || !forecastResponse.ok) {
          throw new Error("Battery analytics request failed");
        }

        const historyData = await historyResponse.json();
        const forecastData = await forecastResponse.json();

        setHistory(historyData.points ?? []);
        setForecast(forecastData.forecast ?? []);
        setAnalyticsMessage("");
      } catch (error) {
        console.error(error);
        setHistory([]);
        setForecast([]);
        setAnalyticsMessage("Could not load analytics for the selected battery.");
      }
    };

    loadBatteryAnalytics();
  }, [selectedBattery]);

  const predictBattery = async () => {
    setLoading(true);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(form),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed");
      }

      const data = await response.json();

      setResult({
        SOH: data.SOH,
        RUL_cycles: data.RUL_cycles,
        anomaly_status: data.anomaly_status,
      });

      const explainResponse = await fetch("/explain", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(form),
      });

      if (explainResponse.ok) {
        const explainData = await explainResponse.json();
        setShapFeatures(explainData.top_features ?? []);
        setExplanation(explainData.interpretation ?? "");
      } else {
        setShapFeatures([]);
        setExplanation("SHAP explanation is unavailable for this prediction.");
      }
    } catch (error) {
      console.error(error);
      alert("Could not connect to FastAPI backend.");
    } finally {
      setLoading(false);
    }
  };

  const uploadBattery = async () => {
    if (!uploadFile) {
      setUploadMessage("Choose a NASA .mat file first.");
      return;
    }

    setUploading(true);
    setUploadMessage("");

    const formData = new FormData();
    formData.append("file", uploadFile);

    try {
      const response = await fetch("/upload-battery", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Upload failed");
      }

      setUploadFile(null);
      setUploadMessage(`Loaded ${data.battery_id} with ${data.cycles} cycles.`);
      await loadBatteries(data.battery_id);
    } catch (error) {
      console.error(error);
      setUploadMessage(
        error instanceof Error
          ? error.message
          : "Could not upload this battery file.",
      );
    } finally {
      setUploading(false);
    }
  };

  return (
    <main className="app">
      <nav className="topbar" aria-label="Application navigation">
        <a className="wordmark" href="#hero" aria-label="EVBI home">
          EVBI
        </a>
        <button
          type="button"
          className="menu-button"
          aria-expanded={navOpen}
          aria-controls="primary-navigation"
          onClick={() => setNavOpen((open) => !open)}
        >
          <span />
          <span />
        </button>
        <div
          id="primary-navigation"
          className={navOpen ? "nav-links open" : "nav-links"}
        >
          <a href="#live-results" onClick={() => setNavOpen(false)}>
            Battery Profile
          </a>
          <a href="#telemetry" onClick={() => setNavOpen(false)}>
            Battery Telemetry
          </a>
          <a href="#analytics" onClick={() => setNavOpen(false)}>
            Battery Health Charts
          </a>
          <a href="#how-it-works" onClick={() => setNavOpen(false)}>
            App Explanation
          </a>
          <a href="#about" onClick={() => setNavOpen(false)}>
            About Us
          </a>
        </div>
      </nav>

      <section id="hero" className="hero-section">
        <div className="hero-copy">
          <div>
            <p className="eyebrow">AI Battery Analytics</p>
            <h1>EV Battery Intelligence</h1>
            <p className="subtitle">
              Predict battery health, remaining useful life and abnormal
              behavior from live battery telemetry using a connected ML
              inference pipeline.
            </p>
          </div>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <div className="battery-visual">
            <span />
            <span />
            <span />
            <span />
          </div>
        </div>
      </section>

      <section id="live-results" className="section live-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Live Results</p>
            <h2>Battery profile</h2>
          </div>
          <span className="connection-pill">API online</span>
        </div>

        <div className="metrics">
          <div className="metric">
            <span>State of health</span>
            <strong>{result.SOH}%</strong>
          </div>

          <div className="metric">
            <span>Remaining useful life</span>
            <strong>{result.RUL_cycles}</strong>
            <small>cycles</small>
          </div>

          <div className="metric">
            <span>Battery status</span>
            <strong>{result.anomaly_status}</strong>
          </div>
        </div>

        <div id="telemetry" className="telemetry-panel">
          <div className="section-heading telemetry-heading">
          <div>
            <p className="eyebrow">Battery Telemetry</p>
            <h2>Enter measurements</h2>
          </div>
          <p className="helper-copy panel-note">
            Adjust the telemetry values and run the saved prediction models.
          </p>
        </div>

          <div className="inputs">
            {fields.map(({ key, label, unit }) => (
              <label key={key}>
                <span>{label}</span>

                <input
                  type="number"
                  step="any"
                  name={key}
                  value={form[key]}
                  onChange={handleChange}
                />

                <small>{unit}</small>
              </label>
            ))}
          </div>

          <button onClick={predictBattery} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Battery"}
          </button>
        </div>
      </section>

      <section id="analytics" className="section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Battery Health</p>
            <h2>Battery analytics charts</h2>
          </div>
          <div className="selector-row">
            <label>
              <span>Battery dataset</span>
              <select
                value={selectedBattery}
                onChange={(event) => setSelectedBattery(event.target.value)}
                disabled={batteryIds.length === 0}
              >
                {batteryIds.length === 0 ? (
                  <option>No battery data found</option>
                ) : (
                  batteryIds.map((batteryId) => (
                    <option key={batteryId} value={batteryId}>
                      {batteryId}
                    </option>
                  ))
                )}
              </select>
            </label>
          </div>
        </div>

        <div className="upload-row">
          <label>
            <span>Upload NASA .mat file</span>
            <input
              type="file"
              accept=".mat"
              onChange={(event) =>
                setUploadFile(event.target.files?.[0] ?? null)
              }
            />
          </label>

          <button
            type="button"
            className="secondary-button"
            onClick={uploadBattery}
            disabled={uploading}
          >
            {uploading ? "Uploading..." : "Upload Battery"}
          </button>
        </div>

        {uploadMessage && <p className="helper-copy">{uploadMessage}</p>}

        {analyticsMessage ? (
          <p className="empty-state">{analyticsMessage}</p>
        ) : (
          <div className="chart-grid">
            <div className="chart-card">
              <span>SOH vs Cycle</span>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="cycle" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="SOH" stroke="#111827" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <span>Capacity degradation vs Cycle</span>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="cycle" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="capacity" stroke="#64748b" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <span>Temperature trends vs Cycle</span>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="cycle" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="temperature" stroke="#b91c1c" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <span>Predicted future SOH degradation</span>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={forecast}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="cycle" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="predicted_SOH"
                    stroke="#2563eb"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
              <small>Estimated with a polynomial degradation baseline.</small>
            </div>
          </div>
        )}
      </section>

      <section id="explainability" className="section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">SHAP Explainability</p>
            <h2>SOH model explanation</h2>
          </div>
          <p className="helper-copy panel-note">
            Positive values lift the SOH prediction. Negative values pull it
            down. These are model contributions, not causal effects.
          </p>
        </div>

        {shapFeatures.length === 0 ? (
          <p className="empty-state">
            Run Analyze Battery to calculate SHAP impacts for the current input.
          </p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Value</th>
                  <th>SHAP impact</th>
                </tr>
              </thead>
              <tbody>
                {shapFeatures.map((feature) => (
                  <tr key={feature.feature}>
                    <td>{feature.feature.replaceAll("_", " ")}</td>
                    <td>{feature.value}</td>
                    <td>
                      {feature.shap_value}{" "}
                      <small className="impact-note">{feature.impact}</small>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {explanation && <p className="helper-copy">{explanation}</p>}
      </section>

      <section id="how-it-works" className="section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">App Explanation</p>
            <h2>How EVBI works</h2>
          </div>
          <p className="helper-copy panel-note">
            The workflow keeps raw telemetry, model inference, anomaly
            detection, charting, and explainability connected in one interface.
          </p>
        </div>
        <div className="pipeline-steps" aria-label="Pipeline process">
          <div>
            <strong>01</strong>
            <span>Collect battery telemetry</span>
          </div>
          <div>
            <strong>02</strong>
            <span>Prepare feature values</span>
          </div>
          <div>
            <strong>03</strong>
            <span>Run saved ML models</span>
          </div>
          <div>
            <strong>04</strong>
            <span>Explain and visualize results</span>
          </div>
        </div>
      </section>

      <section id="about" className="section about-panel">
        <div className="about-hero">
          <p className="eyebrow">About Us</p>
          <h2>How EVBI turns battery data into intelligence</h2>
          <p>
            EVBI is a full-stack EV battery intelligence project that connects a
            React analytics interface to a FastAPI backend and saved machine
            learning models. It accepts live telemetry inputs, supports NASA
            battery history files, displays degradation charts, and explains SOH
            predictions with SHAP values from the real SOH model.
          </p>
        </div>

        <div className="about-grid expanded">
          <article>
            <span>What EVBI is</span>
            <p>
              EVBI focuses on battery monitoring workflows: users enter voltage,
              current, temperature, discharge time, and discharge energy values,
              then receive SOH, RUL, and anomaly outputs from the existing model
              files.
            </p>
          </article>

          <article>
            <span>Why it matters</span>
            <p>
              Battery telemetry becomes useful when it is organized into model
              features, compared with historical degradation behavior, and
              presented with enough explanation for users to understand the
              prediction.
            </p>
          </article>

          <article>
            <span>Implemented models</span>
            <p>
              The backend loads saved pickle files for SOH prediction, RUL
              prediction, anomaly detection, and feature column ordering. The
              app does not retrain or replace these models.
            </p>
          </article>

          <article>
            <span>Future scope</span>
            <p>
              Future work can expand datasets, improve forecasting methods,
              compare more model families, and add richer validation views. The
              current app keeps those future ideas separate from implemented
              behavior.
            </p>
          </article>
        </div>

        <div className="about-subsection">
          <div>
            <p className="eyebrow">Data Processing Pipeline</p>
            <h3>.mat files, CSV files, and preprocessing</h3>
          </div>
          <div className="about-columns pipeline-cards">
            <article>
              <span>01</span>
              <strong>Read NASA files</strong>
              <p>
                Uploaded NASA <code>.mat</code> files are read on the backend
                with <code>scipy.io.loadmat</code>. The parser keeps discharge
                cycles and extracts capacity plus measured temperature values.
              </p>
            </article>
            <article>
              <span>02</span>
              <strong>Normalize history</strong>
              <p>
                Parsed history is normalized into <code>battery_id</code>,
                <code>cycle</code>, <code>capacity</code>, and{" "}
                <code>temperature</code>, then saved under{" "}
                <code>backend/data/processed</code>.
              </p>
            </article>
            <article>
              <span>03</span>
              <strong>Calculate SOH</strong>
              <p>
                Preprocessing converts values to numeric rows, drops invalid
                records, sorts cycles, and calculates SOH from each battery's
                first recorded capacity.
              </p>
            </article>
          </div>
        </div>

        <div className="story-diagram" aria-label="EVBI storytelling architecture diagram">
          <div className="story-intro">
            <p className="eyebrow">Scroll Story Diagram</p>
            <h3>Follow one battery reading through EVBI</h3>
            <p>
              As you scroll, each stage shows how raw telemetry becomes model
              output, explainability, and battery health insight in the
              dashboard.
            </p>
          </div>

          <div className="story-track">
            <article className="story-step">
              <span>01</span>
              <div>
                <strong>Battery data enters</strong>
                <p>Telemetry inputs or NASA history files provide voltage, current, temperature, cycle, and capacity context.</p>
              </div>
            </article>

            <article className="story-step">
              <span>02</span>
              <div>
                <strong>Backend prepares history</strong>
                <p>MAT and CSV processing keeps discharge cycles, extracts capacity and temperature, and calculates SOH from real records.</p>
              </div>
            </article>

            <article className="story-step">
              <span>03</span>
              <div>
                <strong>Features are aligned</strong>
                <p>The API formats telemetry into the saved `feature_columns.pkl` order before running the existing model files.</p>
              </div>
            </article>

            <article className="story-step">
              <span>04</span>
              <div>
                <strong>Models make predictions</strong>
                <p>SOH, RUL, and anomaly models produce the live battery profile without retraining or replacing the saved ML logic.</p>
              </div>
            </article>

            <article className="story-step">
              <span>05</span>
              <div>
                <strong>SHAP explains the result</strong>
                <p>Feature contributions show what pushed this prediction higher or lower, without treating those values as causal effects.</p>
              </div>
            </article>

            <article className="story-step">
              <span>06</span>
              <div>
                <strong>React tells the story</strong>
                <p>The dashboard updates cards, charts, forecast lines, and explanation tables so users can inspect the battery state.</p>
              </div>
            </article>
          </div>
        </div>

        <div className="about-subsection final-about-grid">
          <article>
            <span>01</span>
            <span>Machine learning approach</span>
            <p>
              <code>/predict</code> creates a one-row Pandas DataFrame using
              the saved <code>feature_columns.pkl</code> order, then calls the
              existing SOH, RUL, and anomaly model files.
            </p>
          </article>
          <article>
            <span>02</span>
            <span>FastAPI integration</span>
            <p>
              FastAPI exposes <code>/predict</code>, <code>/explain</code>,
              <code>/batteries</code>, history, forecast, and upload endpoints
              to connect the frontend to models and processed history data.
            </p>
          </article>
          <article>
            <span>03</span>
            <span>Frontend dashboard</span>
            <p>
              React and Recharts render the Battery Profile, Battery Telemetry
              form, NASA battery charts, forecast line, upload control, and SHAP
              explanation table without changing backend behavior.
            </p>
          </article>
        </div>
      </section>
    </main>
  );
}

export default App;
