# MPEA Generative Design – Local Streamlit App

Based on Li & Birbilis (2024):
"Multi-objective Optimization-Oriented Generative Adversarial Design for Multi-principal Element Alloys"

---

## Folder Structure (what you need)

```
mpea_app/
├── app.py                  ← main Streamlit application (this file)
├── requirements.txt        ← Python dependencies
└── models/                 ← all pre-trained model files (copy from repo)
    ├── generator_net_MPEA.pt
    ├── tensile_regressor.joblib
    ├── elongation_regressor.joblib
    ├── yield_regressor.joblib
    ├── hardness_regressor.joblib
    ├── FCC_classifier.joblib
    ├── BCC_classifier.joblib
    ├── HCP_classifier.joblib
    ├── IM_classifier.joblib
    └── MPEA_parsed_dataset.xlsx
```

All model files come from the `streamlit/` folder in the original GitHub repo.

---

## Step-by-Step Setup

### Step 1 – Install Python 3.10
Download from https://www.python.org/downloads/
Verify: `python --version`  →  should say Python 3.10.x

### Step 2 – Create a virtual environment
```bash
# Navigate into the app folder
cd mpea_app

# Create venv
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```
This installs: streamlit, torch, pymoo, scikit-learn, pandas, matplotlib, etc.

### Step 4 – Run the app
```bash
streamlit run app.py
```
The browser will open automatically at: http://localhost:8501

---

## How to Use the App

1. In the **left sidebar**, select your optimisation objectives
   (e.g. Tensile Strength + Elongation)
2. Adjust Population Size and Number of Generations
   - More generations = better results but slower (~15–60 seconds)
3. Click **Start Optimisation**
4. View the scatter plots and the results table
5. Click **Download Results as Excel** to save

---

## Objectives Explained

| Objective        | Effect                          |
|------------------|---------------------------------|
| Tensile Strength | Maximise UTS (MPa)              |
| Elongation       | Maximise ductility (%)          |
| Yield Strength   | Maximise yield point (MPa)      |
| Hardness         | Maximise hardness (HV)          |
| Density          | **Minimise** density (g/cm³)    |
| FCC / BCC / HCP / IM | Maximise probability of that crystal phase |
| Aluminum Content | Maximise Al molar ratio         |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: pymoo` | `pip install pymoo==0.6.0.1` |
| `torch` fails to install | Visit https://pytorch.org and use their install selector |
| `No such file: models/...` | Make sure the `models/` folder is inside `mpea_app/` |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Scikit-learn version error | `pip install scikit-learn>=1.3.0` |
