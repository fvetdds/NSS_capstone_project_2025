import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import xgboost as xgb

# Streamlit page config
st.set_page_config(page_title="Breast Cancer Risk Prediction", layout="wide")

# --- CUSTOM HEADER ---
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

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["About", "Risk Insights", "Mind & Move"])

# --- LOAD MODEL & THRESHOLD ---
BASE_DIR = Path(__file__).resolve().parent

# Load raw XGBoost Booster
booster = xgb.Booster()
booster.load_model(str(BASE_DIR / "models" / "bcsc_xgb_model.bst"))

# Load decision threshold
threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")

# --- TAB 1: ABOUT ---
with tab1:
    st.markdown("### 📊 About the Data and Model Behind this Risk Factor Prediction")
    st.markdown("""
XGBoost is a powerful tree-based model for tabular data, capable of capturing complex feature interactions.  
We trained it to prioritize detecting true cancer cases, achieving 89% accuracy on held-out data.  
Our dataset is the Breast Cancer Surveillance Consortium (BCSC) cohort, with millions of mammogram records and outcomes.  
[Learn more about BCSC](https://www.bcsc-research.org/)

**Key Test Metrics**  
- Sensitivity (Recall) for cancer: 52%  
- Specificity for non-cancer: 92%  
- Precision for cancer predictions: 52%  
- ROC AUC: 0.91  
- Threshold used: 0.82  
- Matthews Correlation Coefficient: 0.5573
""")
    st.markdown("### BCSC Data Details")
    st.image("figures/age_group_label_by_cancer_label.png", width=900)
    st.markdown("Most participants are aged 45–74, with peaks in 50–59.")
    st.image("figures/bmi_group_label_by_cancer_label.png", width=900)
    st.markdown("BMI is concentrated in 10–24.9 and 25–29.9 groups.")
    st.image("figures/first_degree_hx_label_by_cancer_label.png", width=900)
    st.markdown("Family history is more common among those with cancer.")
    st.image("figures/feature_importance_xgb.png", width=900)
    st.markdown("**Top Features:** Age group, BI-RADS density, family history, etc.")

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

    # Align features to booster expectation
    expected = booster.feature_names
    df_new = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)

    # Predict with DMatrix
    dmat = xgb.DMatrix(df_new)
    prob = booster.predict(dmat)[0]

    risk_str = "High risk" if prob >= threshold else "Low risk"
    icon     = "⚠️" if risk_str == "High risk" else "✅"

    st.subheader("Breast Cancer Risk Prediction")
    st.write(f"Predicted probability of breast cancer: {prob:.1%}")
    if risk_str == "High risk":
        st.error(f"{icon} {risk_str} (threshold = {threshold:.2f})")
    else:
        st.success(f"{icon} {risk_str} (threshold = {threshold:.2f})")

# --- TAB 3: MIND & MOVE ---
# Daily goals
MEDITATE_GOAL = 10   # minutes
EXERCISE_GOAL = 30   # minutes
WATER_GOAL    = 8    # glasses

with tab3:
    st.header("Glow and Grow")
    st.write("Every healthy choice you make today is a step toward a brighter, happier you. You’ve got this!")

    st.subheader("Shape The Future U Tracker")
    c1, c2, c3 = st.columns(3)
    with c1:
        meditate = st.number_input("Meditation minutes", 0, 60, 0)
        st.progress(meditate / MEDITATE_GOAL)
        st.metric("Meditation", f"{meditate} min", f"{max(0, MEDITATE_GOAL-meditate)} to goal")

    with c2:
        exercise = st.number_input("Exercise minutes", 0, 120, 0)
        st.progress(exercise / EXERCISE_GOAL)
        st.metric("Exercise", f"{exercise} min", f"{max(0, EXERCISE_GOAL-exercise)} to goal")

    with c3:
        water = st.number_input("Glasses of water", 0, 20, 0)
        st.progress(water / WATER_GOAL)
        st.metric("Hydration", f"{water} glasses", f"{max(0, WATER_GOAL-water)} to goal")

    st.subheader("Diet Log")
    diet = st.text_area("Meals / snacks")
    if st.button("Save Entry"):
        entry = {
            "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "meditation": meditate,
            "exercise": exercise,
            "water": water,
            "diet": diet
        }
        st.success("Your daily wellness entry has been recorded!")
        st.json(entry)

    st.subheader("Additional Resources")
    st.markdown("**YouTube Videos:**")
    videos = {
        "Mindfulness Meditation for Cancer Support": "https://www.youtube.com/watch?v=1ZYbU82GVz4&t=31s",
        "Gentle Move for All": "https://www.youtube.com/watch?v=Ev6yE55kYGw&t=169s",
        "Healthy Eating During Cancer Treatment": "https://www.youtube.com/shorts/kkk8UPd7l38"
    }
    for title, url in videos.items():
        st.markdown(f"- [{title}]({url})")

    st.markdown("**Local Support Groups in Nashville, TN:**")
    support_groups = [
        {"name":"Susan G. Komen Nashville",           "phone":"(615) 673-6633", "website":"https://komen.org/nashville"},
        {"name":"Vanderbilt Breast Cancer Support",   "phone":"(615) 322-3900", "website":"https://www.vicc.org/support-groups"},
        {"name":"Alive Hospice Cancer Support",       "phone":"(615) 327-1085", "website":"https://alivehospice.org"},
    ]
    for grp in support_groups:
        st.markdown(f"- **{grp['name']}**: {grp['phone']} | [Website]({grp['website']})")
