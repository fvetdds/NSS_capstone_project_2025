import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import types
from pathlib import Path

# ─── Reconstruct custom loss for un-pickling ───────────────────────────────────
# Replace scale_pos_weight with the exact value from your training script
scale_pos_weight   =  len([0])  # placeholder, replace with len(neg)/len(pos) from model_train.py
pos_penalty_factor = 10.0
weight_neg         = 1.0
weight_pos         = scale_pos_weight * pos_penalty_factor

def weighted_logloss(y_true: np.ndarray, y_pred: np.ndarray):
    p    = 1.0 / (1.0 + np.exp(-y_pred))
    w    = np.where(y_true == 1, weight_pos, weight_neg)
    grad = w * (p - y_true)
    hess = w * p * (1.0 - p)
    return grad, hess

# Inject fake module so pickle finds weighted_logloss in "model_train"
fake_mod = types.ModuleType("model_train")
fake_mod.weighted_logloss = weighted_logloss
sys.modules["model_train"] = fake_mod

# ─── Streamlit page setup ─────────────────────────────────────────────────────
st.set_page_config(page_title="Breast Cancer Risk Prediction", layout="wide")

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
tab1, tab2, tab3 = st.tabs(["About", "Risk Insights", "Mind & Move"])

# ─── Load model & threshold ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
model     = joblib.load(BASE_DIR / "models" / "bcsc_xgb_model.pkl")
threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")

# ─── TAB 1: ABOUT ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 About the Data and Model Behind this Risk Factor Prediction")
    st.markdown("""
XGBoost is a state-of-the-art tree-based model for tabular data, capturing complex feature interactions.  
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

# ─── TAB 2: RISK INSIGHTS ───────────────────────────────────────────────────────
with tab2:
    with st.expander("Your information for risk prediction", expanded=True):
        def sel(label, opts):
            return st.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])

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
            "age_group":         sel("Age group", age_groups),
            "race_eth":          sel("Race/Ethnicity", race_eth),
            "age_menarche":      sel("Age at 1st period", menarche),
            "age_first_birth":   sel("Age at first birth", birth_age),
            "family_history":    sel("Family history of cancer", fam_hist),
            "personal_biopsy":   sel("Personal biopsy history", biopsy),
            "density":           sel("BI-RADS density", density),
            "hormone_use":       sel("Hormone use", hormone_use),
            "menopausal_status": sel("Menopausal status", menopause),
            "bmi_group":         sel("BMI group", bmi_group),
        }

    raw_df = pd.DataFrame(inputs, index=[0])
    expected = model.get_booster().feature_names
    df_new   = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)

    prob     = model.predict_proba(df_new)[0, 1]
    risk_str = "High risk" if prob >= threshold else "Low risk"
    icon     = "⚠️" if risk_str == "High risk" else "✅"

    st.subheader("Breast Cancer Risk Prediction")
    st.write(f"Predicted probability of breast cancer: {prob:.1%}")
    if risk_str == "High risk":
        st.error(f"{icon} {risk_str} (threshold = {threshold:.2f})")
    else:
        st.success(f"{icon} {risk_str} (threshold = {threshold:.2f})")

# ─── TAB 3: MIND & MOVE ─────────────────────────────────────────────────────────
with tab3:
    st.header("Glow and Grow")
    st.write("Every healthy choice you make today is a step toward a brighter, happier you. You’ve got this!")
    st.subheader("Daily Rituals")
    tips = [
        "🧘 Practice 10 minutes of mindfulness meditation",
        "🥗 Include at least 5 servings of fruits and vegetables",
        "🚶‍♀️ Take a 30-minute brisk walk or light exercise",
        "💧 Stay hydrated by drinking 8 glasses of water",
        "😴 Aim for 7-8 hours of sleep each night"
    ]
    for tip in tips:
        st.markdown(f"- {tip}")

    st.subheader("Shape The Future U Tracker")
    col1, col2, col3 = st.columns(3)
    with col1:
        meditate_mins = st.number_input("Meditation minutes", 0, 60, 0)
    with col2:
        exercise_mins = st.number_input("Exercise minutes", 0, 180, 0)
    with col3:
        water_glasses = st.number_input("Glasses of water", 0, 20, 0)

    if st.button("Save Entry"):
        entry = {
            "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "meditation": meditate_mins,
            "exercise": exercise_mins,
            "water": water_glasses
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
        {"name":"Susan G. Komen Nashville",         "phone":"(615) 673-6633", "website":"https://komen.org/nashville"},
        {"name":"Vanderbilt Breast Cancer Support", "phone":"(615) 322-3900", "website":"https://www.vicc.org/support-groups"},
        {"name":"Alive Hospice Cancer Support",     "phone":"(615) 327-1085", "website":"https://alivehospice.org"},
    ]
    for grp in support_groups:
        st.markdown(f"- **{grp['name']}**: {grp['phone']} | [Website]({grp['website']})")
