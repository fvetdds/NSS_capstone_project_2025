import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Breast Cancer Risk prediction", layout="wide")

# ---- Custom Navbar CSS ----
st.markdown("""
    <style>
    .navbar-container {
        background: #0B1446;
        border-radius: 0 0 18px 18px;
        margin-bottom: 1.2em;
        width: 100vw;
        min-width: 100vw;
        padding: 0.7em 0 0.7em 0;
        position: relative;
        left: -2.7vw;
        z-index: 100;
        display: flex;
        align-items: center;
    }
    .navbar-logo {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFD700;
        display: flex;
        align-items: center;
        white-space: nowrap;
        margin-left: 2.4em;
        margin-right: 2.5em;
    }
    .navbar-logo span {
        font-size: 2.2rem;
        margin-right: 0.6em;
    }
    .nav-btn {
        background: transparent;
        border: none;
        color: #FFD700;
        font-size: 1.35rem;
        font-weight: 800;
        padding: 0.38em 2.2em 0.38em 2.2em;
        border-radius: 9px;
        margin-right: 1.7em;
        transition: color 0.22s, background 0.19s;
        outline: none;
        cursor: pointer;
    }
    .nav-btn.selected, .nav-btn:active {
        color: #fff700 !important;
        background: #19225c !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Tabs and State ----
tabs = ["About", "Risk Insights", "Mind & Move"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = tabs[0]

# ---- Draw Custom Navbar ----
cols = st.columns([2, 1.3, 1.7, 1.7, 8])
with cols[0]:
    st.markdown('<div class="navbar-logo"><span>🎗️</span> EmpowerHER</div>', unsafe_allow_html=True)
for i, t in enumerate(tabs):
    with cols[i+1]:
        is_selected = st.session_state.active_tab == t
        btn_class = "nav-btn selected" if is_selected else "nav-btn"
        btn_clicked = st.button(
            t,
            key=f"nav_{t}",
            help=f"Go to {t}",
            use_container_width=True
        )
        # CSS for selected tab
        st.markdown(
            f"""<style>
            [data-testid="stButton"] button#nav_{t} {{
                background: transparent !important;
                color: {'#fff700' if is_selected else '#FFD700'} !important;
                font-size: 1.35rem !important;
                font-weight: 800 !important;
                border-radius: 9px !important;
                padding: 0.38em 2.2em 0.38em 2.2em !important;
                margin-right: 1.7em !important;
                background: {'#19225c' if is_selected else 'transparent'} !important;
            }}
            </style>
            """, unsafe_allow_html=True
        )
        if btn_clicked:
            st.session_state.active_tab = t

st.markdown("---")

# ---- Load models and data (only once) ----
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models" / "bcsc_xgb_model.pkl")
threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")

tab = st.session_state.active_tab

# ---- Render Content for Each Tab ----
if tab == "About":
    st.markdown("### 📊 About the Data and Model Behind this Risk Factor Prediction")
    st.markdown("""
XGBoost machine learning model is one of the best for tabular data and can handle complex relationship nd interactions between features. We build a model that prioritized finding as many true cancer cases as possible--it catches almost 9 out of 10 cases in test dataset. This trained XGBoost model predicits the likelihood that someone has or will have breast cancer based on their health and demographic data. The train and test dataset is the Breast Cancer Surveillance Consortium (BCSC) dataset contains millions of mammogram records, risk factors, and cancer outcomes from diverse populations in the U.S.  [Learn more about BCSC](https://www.bcsc-research.org/).
After each person entered demographic and medical information, the model gives a probability score to predict what the chance this person will have cancer.
How Accurate is the Model?
- Recall for Cancer is 89%. it finds 89% of those who actually do have cancer in the test dataset, while recall for No cancer is 77%. The model has ROC AUC:0.91, threshold of 0.53 and Matthews Correlation Coefficient of 0.5.
""")
    st.image("figures/age_group_label_by_cancer_label.png", width=900)
    st.markdown("""The majority of study participants fall in the 45–74 age range, with the highest counts in the 50–59 and 55–59 age groups.""")
    st.image("figures/bmi_group_label_by_cancer_label.png", width=900)
    st.markdown("""The largest number of participants, both with and without breast cancer history, are in the lower BMI groups (10–24.99 and 25–29.99).""")
    st.image("figures/first_degree_hx_label_by_cancer_label.png", width=900)
    st.markdown("""Most participants do not have a first-degree family history of breast cancer, regardless of their own cancer history. However, among those with a history of breast cancer (orange bars), a larger proportion report a family history of the disease compared to those without cancer.""")
    st.image("figures/feature_importance_xgb.png", width=900)
    st.markdown("""**Which Factors Matter Most?**  
    The feature importance plot shows which risk factors contribute most to the model's predictions.""")

elif tab == "Risk Insights":
    with st.expander("Your information for risk prediction", expanded=True):
        def sel(label, opts):
            return st.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])
        age_groups  = {1:"18–29", 2:"30–34", 3:"35–39", 4:"40–44", 5:"45–49", 6:"50–54", 7:"55–59", 8:"60–64", 9:"65–69", 10:"70–74", 11:"75–79", 12:"80–84", 13:">85"}
        race_eth    = {1:"White", 2:"Black", 3:"Asian or Pacific Island", 4:"Native American", 5:"Hispanic", 6:"Other"}
        menarche    = {0:">14", 1:"12–13", 2:"<12"}
        birth_age   = {0:"<20", 1:"20–24", 2:"25–29", 3:">30", 4:"Nulliparous"}
        fam_hist    = {0:"No", 1:"Yes"}
        biopsy      = {0:"No", 1:"Yes"}
        density     = {1:"Almost fat", 2:"Scattered", 3:"Hetero-dense", 4:"Extremely"}
        hormone_use = {0:"No", 1:"Yes"}
        menopause   = {1:"Pre/peri", 2:"Post", 3:"Surgical"}
        bmi_group   = {1:"10–24.9", 2:"25–29.9", 3:"30–34.9", 4:"35+"}
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
    df_new = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)
    prob = model.predict_proba(df_new)[0, 1]
    risk_str = "High risk" if prob >= threshold else "Low risk"
    icon = "⚠️" if risk_str == "High risk" else "✅"
    st.subheader("Breast Cancer Risk Prediction")
    st.write(f"Predicted probability of breast cancer: {prob:.1%}")
    if risk_str == "High risk":
        st.error(f"{icon} {risk_str} (threshold = {threshold:.2f})")
    else:
        st.success(f"{icon} {risk_str} (threshold = {threshold:.2f})")

elif tab == "Mind & Move":
    st.header("Glow and Grow")
    st.write("Here are some tips and a simple tracker to help you with meditation, diet, and exercise.")
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
        meditate_mins = st.number_input("Meditation minutes", min_value=0, max_value=60, value=0)
    with col2:
        exercise_mins = st.number_input("Exercise minutes", min_value=0, max_value=180, value=0)
    with col3:
        water_glasses = st.number_input("Glasses of water", min_value=0, max_value=20, value=0)
    diet_log = st.text_area("Diet log (meals/snacks)")
    if st.button("Save Entry"):
        entry = {
            "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "meditation": meditate_mins,
            "exercise": exercise_mins,
            "water": water_glasses,
            "diet": diet_log
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
        {
            "name": "Susan G. Komen Nashville",
            "phone": "(615) 673-6633",
            "website": "https://komen.org/nashville"
        },
        {
            "name": "Vanderbilt Breast Cancer Support Group",
            "phone": "(615) 322-3900",
            "website": "https://www.vicc.org/support-groups"
        },
        {
            "name": "Alive Hospice Cancer Support",
            "phone": "(615) 327-1085",
            "website": "https://alivehospice.org"
        }
    ]
    for grp in support_groups:
        st.markdown(f"- **{grp['name']}**: {grp['phone']} | [Website]({grp['website']})")
