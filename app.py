import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import xgboost as xgb
import os

# Streamlit page config
st.set_page_config(page_title="Breast Cancer Risk Prediction", layout="wide")

# HEADER 
st.markdown("""
    <style>
    .navbar-logo {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFD700;
        display: flex;
        align-items: center;
        white-space: nowrap;
        margin-left: 2.4em;
        margin-top: 1.2em;
        margin-bottom: 0.2em;
    }
    .navbar-logo span {
        font-size: 2.2rem;
        margin-right: 0.6em;
    }
    </style>
    <div class="navbar-logo"><span>🎗️</span> EmpowerHER</div>
""", unsafe_allow_html=True)
st.markdown("---")

# TABS details
tab1, tab2, tab3 = st.tabs(["About", "Risk Insights", "Mind & Move"])

# LOAD MODEL & THRESHOLD ---
BASE_DIR = Path(__file__).resolve().parent
models_dir = BASE_DIR / "models"


st.write("📦 models directory:", os.listdir(models_dir))

bst_path = models_dir / "bcsc_xgb_model.bst"
if not bst_path.exists() or bst_path.stat().st_size == 0:
    st.error(f"Model file missing or empty: {bst_path.name}")
    st.stop()


model = xgb.XGBClassifier()
# 2) Load the saved Booster into it
model.load_model(str(bst_path))

threshold = joblib.load(models_dir / "threshold.pkl")

# --- TAB 1: ABOUT ---
with tab1:
    st.markdown("### 📊 About the Data and Model Behind this Risk Factor Prediction")
    st.markdown("""
XGBoost is a powerful tree-based model for tabular data, capable of capturing complex feature interactions.  
We trained it to prioritize detecting true cancer cases, achieving 89% accuracy on held-out data.  
Our dataset is the Breast Cancer Surveillance Consortium (BCSC) cohort, with millions of mammogram records and outcomes.  
[Learn more about BCSC](https://www.bcsc-research.org/)

**Key Test Metrics**  
- Sensitivity (Recall) for cancer: 54%  
- Specificity for non-cancer: 92%  
- Precision for cancer predictions: 70%  
- ROC AUC: 0.91  
- Threshold used: 0.82  
- Matthews Correlation Coefficient: 0.5573
""")
    
# --- TAB 2: RISK INSIGHTS ---
with tab2:
    with st.expander("Your information for risk prediction", expanded=True):
        def sel(label, options):
            return st.selectbox(label, list(options.keys()), format_func=lambda k: options[k])

        age_groups  = {1:"18–29",2:"30–34",3:"35–39",4:"40–44",5:"45–49",
                       6:"50–54",7:"55–59",8:"60–64",9:"65–69",10:"70–74",
                       11:"75–79",12:"80–84",13:">85"}
        race_eth    = {1:"White",2:"Black",3:"Asian/Pacific Island",4:"Native American",
                       5:"Hispanic",6:"Other"}
        menarche    = {0:">14",1:"12–13",2:"<12"}
        birth_age   = {0:"<20",1:"20–24",2:"25–29",3:">30",4:"Nulliparous"}
        fam_hist    = {0:"No",1:"Yes"}
        biopsy      = {0:"No",1:"Yes"}
        density     = {1:"Almost fat",2:"Scattered",3:"Hetero-dense",4:"Extremely dense"}
        hormone_use = {0:"No",1:"Yes"}
        menopause   = {1:"Pre/peri",2:"Post",3:"Surgical"}
        bmi_group   = {1:"10–24.9",2:"25–29.9",3:"30–34.9",4:"35+"}

        inputs = {
            "age_group":       sel("Age group", age_groups),
            "race_eth":        sel("Race/Ethnicity", race_eth),
            "age_menarche":    sel("Age at 1st period", menarche),
            "age_first_birth": sel("Age at first birth", birth_age),
            "family_history":  sel("Family history of cancer", fam_hist),
            "personal_biopsy": sel("Personal biopsy history", biopsy),
            "density":         sel("BI-RADS density", density),
            "hormone_use":     sel("Hormone use", hormone_use),
            "menopausal_status": sel("Menopausal status", menopause),
            "bmi_group":       sel("BMI group", bmi_group),
        }

    raw_df = pd.DataFrame(inputs, index=[0])

    # Align features
    expected = model.get_booster().feature_names
    df_new = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)

    # Predict probability
    prob = model.predict_proba(df_new)[0, 1]

    risk_str = "High risk" if prob >= threshold else "Low risk"
    icon     = "⚠️" if risk_str == "High risk" else "✅"

    st.subheader("Breast Cancer Risk Prediction")
    st.write(f"Predicted probability of breast cancer: {prob:.1%}")
    if risk_str == "High risk":
        st.error(f"{icon} {risk_str} (threshold = {threshold:.2f})")
    else:
        st.success(f"{icon} {risk_str} (threshold = {threshold:.2f})")

# --- TAB 3: MIND & MOVE ---
MEDITATE_GOAL = 10
EXERCISE_GOAL = 30
WATER_GOAL    = 8

with tab3:
    st.header("Glow and Grow")
    st.write("Every healthy choice you make today is a step toward a brighter, happier you. You’ve got this!")
    # ... (rest of your Mind & Move tracker) ...
