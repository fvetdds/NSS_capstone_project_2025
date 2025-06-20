import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import types
from pathlib import Path

#define weighted_logloss

def weighted_logloss(y_true: np.ndarray, y_pred: np.ndarray):
    grad = np.zeros_like(y_pred)
    hess = np.zeros_like(y_pred)
    return grad, hess

# add in the original training module name:
mod = types.ModuleType("model_train")
mod.weighted_logloss = weighted_logloss
sys.modules["model_train"] = mod

# Streamlit page setup 
st.set_page_config(page_title="Breast Cancer Risk Prediction", layout="wide")

st.markdown("""
    <style>
    .navbar-logo {
        font-size: 2.5rem; font-weight: bold; color: #FFD700;
        display: flex; align-items: center; white-space: nowrap;
        margin-left: 2.4em; margin-top: 1.2em; margin-bottom: 0.2em;
    }
    .navbar-logo span { font-size: 2.2rem; margin-right: 0.6em; }
    </style>
    <div class="navbar-logo"><span>🎗️</span> EmpowerHER</div>
""", unsafe_allow_html=True)
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["About", "Breast Cancer Risk Prediction", "Mind & Move"])

# ─── 3) Load your pickled model & threshold ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
model     = joblib.load(BASE_DIR / "models" / "bcsc_xgb_model.pkl")
threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")

# ─── 4) Tab 1: About ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 About this Breast Cancer Risk Model")
    st.markdown("""
**XGBoost** is a state-of-the-art tree-based model for tabular data, capturing complex feature interactions.  
I trained it on the Brest Cancer Surveillance Consortium **BCSC** cohort https://www.bcsc-research.org/index.php/datasets/rf/documentation with a custom weighted log-loss to prioritize cancer detection,  
achieving **89% overall accuracy**, **52% recall** on true cancer cases, **Matthews Correlation Coefficient: 0.54** and **0.92 ROC-AUC**.  
We use a **0.98** probability threshold to flag “High risk.”  
""")
    st.image("figures/empowerher_risk_pipeline_clean.png", width=900)
    st.markdown("Users can select demographic and clinical data to see the model risk prediction.")
    st.image("figures/feature_importance_xgb.png", width=900)
    st.markdown("This plot shows the top predictors the model relies on.")
    st.image("figures/P-R chart2.png", width=900)
    st.markdown("Precision–Recall curve for this XGBoost classifier model.")
# ─── 5) Tab 2: Risk Insights ────────────────────────────────────────────────────
with tab2:
    with st.expander("Enter your details for risk prediction", expanded=True):
        def sel(label, opts):
            return st.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])
        age_groups  = {1:"18–29",2:"30–34",3:"35–39",4:"40–44",5:"45–49",6:"50–54",
                       7:"55–59",8:"60–64",9:"65–69",10:"70–74",11:"75–79",12:"80–84",13:">85"}
        race_eth    = {1:"White",2:"Black",3:"Asian/Pacific",4:"Native",5:"Hispanic",6:"Other"}
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
            "family_history":    sel("Family history", fam_hist),
            "personal_biopsy":   sel("Personal biopsy history", biopsy),
            "density":           sel("BI-RADS density", density),
            "hormone_use":       sel("Hormone use", hormone_use),
            "menopausal_status": sel("Menopausal status", menopause),
            "bmi_group":         sel("BMI group", bmi_group),
        }

    raw_df   = pd.DataFrame(inputs, index=[0])
    expected = model.get_booster().feature_names
    df_new   = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)
    prob     = model.predict_proba(df_new)[0, 1]

    risk_str = "High risk" if prob >= threshold else "Low risk"
    icon     = "⚠️" if risk_str=="High risk" else "✅"

    st.subheader("Your Predicted Risk")
    st.write(f"Probability of breast cancer: **{prob:.1%}**")
    if risk_str=="High risk":
        st.error(f"{icon} {risk_str}  (threshold = {threshold:.2f})")
    else:
        st.success(f"{icon} {risk_str}  (threshold = {threshold:.2f})")

# ─── 6) Tab 3: Mind & Move ─────────────────────────────────────────────────────
with tab3:
    st.header("Glow and Grow")
    st.write("Every healthy choice you make today is a step toward a brighter, happier you. You’ve got this!")
    st.subheader("Daily Rituals")
    for tip in [
        "🧘 Practice 10 min mindfulness",
        "🥗 Eat ≥5 servings fruits/veggies",
        "🚶‍♀️ Take a 30 min walk",
        "💧 Drink 8 glasses of water",
        "😴 Get 7–8 h sleep"
    ]:
        st.markdown(f"- {tip}")
    st.subheader("Tracker")
    c1, c2, c3 = st.columns(3)
    with c1:
        med = st.number_input("Meditation (min)", min_value=0, max_value=60, value=0)
        st.progress(med/10)
        st.metric("Meditation", f"{med} min", f"{10-med} to goal")
    with c2:
        ex = st.number_input("Exercise (min)", min_value=0, max_value=180, value=0)
        st.progress(ex/30)
        st.metric("Exercise", f"{ex} min", f"{30-ex} to goal")
    with c3:
        water = st.number_input("Water (glasses)", min_value=0, max_value=20, value=0)
        st.progress(water/8)
        st.metric("Hydration", f"{water} glasses", f"{8-water} to goal")
    if st.button("Save Entry"):
        st.success("Your daily wellness entry has been recorded!")  
        st.json(entry)
    

    st.title("Additional Resources")
    videos = {
    "Mindfulness Meditation for Cancer Support": "https://www.youtube.com/watch?v=1ZYbU82GVz4&t=31s",
    "Gentle Move for All":                     "https://www.youtube.com/watch?v=Ev6yE55kYGw&t=169s",
    "Healthy Eating During Cancer Treatment":   "https://www.youtube.com/shorts/zci6v7TINa4"
}

    st.markdown("## 🎥 YouTube Videos")
    cols = st.columns(len(videos))
    for col, (title, url) in zip(cols, videos.items()):
        col.video(url)
        col.caption(f"**{title}**")


    st.markdown("## 🤝 Local Support Groups")
    support = [
      {"Name": "American Cancer Society Middle Tennessee",            "Phone": "(800) 227-2345 (toll free)", "Website": "https://www.cancer.org/about-us/local/tennessee.html"},
     {"Name": "Cancer Support Community Middle Tennessee",            "Phone": "(615) 329-5266", "Website": "https://cancersupportcommunity.org/middletn"},
    {"Name": "Susan G. Komen Nashville",            "Phone": "(615) 673-6633", "Website": "https://komen.org/nashville"},
    {"Name": "Vanderbilt Breast Cancer Support Group","Phone": "(615) 322-3900", "Website": "https://www.vicc.org/support-groups"},
    {"Name": "Alive Hospice Cancer Support",        "Phone": "(615) 327-1085", "Website": "https://alivehospice.org"},
    {"Name": "YMCA of Middle Tennessee – LIVESTRONG® at the Y", "Website": "https://ymcamidtn.org/program/livestrong"}
]
support_df = pd.DataFrame(support)
support_df["Website"] = support_df["Website"].apply(lambda url: f"[Visit]({url})")
st.table(supprot_df)