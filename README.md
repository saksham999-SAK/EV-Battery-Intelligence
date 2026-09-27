# EV Battery Intelligence

Machine-learning system for lithium-ion battery health analysis.

## Outputs

- State of Health (SOH)
- Remaining Useful Life (RUL)
- Anomaly status

## Project structure

```text
EV-Battery-Intelligence/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── soh_model.pkl
│   ├── rul_model.pkl
│   ├── anomaly_model.pkl
│   └── feature_columns.pkl
└── frontend/
```

## Important

The `.pkl` model files are generated in your Google Colab notebook. Copy/download those four files from Colab into `backend/` before starting the API.

## Run backend

```bash
cd backend
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start API:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```
