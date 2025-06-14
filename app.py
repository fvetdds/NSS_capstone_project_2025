import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page config
st.set_page_config(page_title="Breast Cancer Risk & Survival", layout="wide")
st.markdown("""
<style>
/* Tab row: subtle thin line (gold) under all tabs */
.stTabs [role="tablist"] {
    border-bottom: 2.5px solid #FFD700 !important;   /* gold underline */
    margin-bottom: 1.2em;
    padding-left: 0.5em;
    padding-right: 0.5em;
}

/* Tabs: no background, just floating text with spacing */
.stTabs [data-baseweb="tab"] {
    font-size: 1.22rem;
    font-weight: 700;
    color: #FFD700;
    background: transparent !important;
    margin-right: 2.5em !important;  /* space between tabs */
    border: none !important;
    box-shadow: none !important;
    padding: 0.65em 0.8em 0.65em 0.8em;
    border-radius: 0;
    transition: color 0.2s, border-bottom 0.2s;
}

/* Only active tab gets the underline (thicker and gold) */
.stTabs [aria-selected="true"] {
    color: #FFD700 !important;
    border-bottom: 4px solid #FFD700 !important;
    font-weight: 900;
    background: transparent !important;
}

/* Optional: hover effect */
.stTabs [data-baseweb="tab"]:hover {
    color: #fff700 !important;   /* slightly lighter yellow */
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ---- Custom header: title + images + empty for tabs ----
st.markdown(
    """
    <div class="header-row">
        <div class="header-logo">🎗️ EmpowerHER</div>
        <div class="tabs-container"></div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Add tab highlight CSS after header
st.markdown("""
<style>
.stTabs [data-baseweb="tab"] {
    font-size: 1.2rem;
    color: #008080;  /* teal */
    background: #FFD700;
    border-bottom: 2px solid transparent;
    transition: border-color 0.3s;
}
.stTabs [aria-selected="true"] {
    font-weight: bold;
    color: #fff !important;     /* white text */
    border-bottom: 4px solid #ffd700 !important;  /* thick gold underline */
    background: #232323;        /* slight dark bg */
}
.stTabs [role="tablist"] {
    border-bottom: 1px solid #ffd700;
    margin-bottom: 2em;
}
</style>
""", unsafe_allow_html=True)

# Load models and data
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models" / "bcsc_xgb_model.pkl")
threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")

# Create tabs
tab1, tab2 = st.tabs(["Risk Insights", "Mind & Move"])

# --- Tab 1: Breast Cancer Risk Predictor ---
with tab1:
    with st.expander("Your information for risk prediction", expanded=True):
        def sel(label, opts):
            return st.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])

        # Define dropdown options
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

        # Collect inputs
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

    # Prepare DataFrame for prediction
    raw_df = pd.DataFrame(inputs, index=[0])
    expected = model.get_booster().feature_names
    df_new = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)

    # Predict probability
    prob = model.predict_proba(df_new)[0, 1]
    risk_str = "High risk" if prob >= threshold else "Low risk"
    icon = "⚠️" if risk_str == "High risk" else "✅"

    # Display results
    st.subheader("Breast Cancer Risk Prediction")
    st.write(f"Predicted probability of breast cancer: {prob:.1%}")
    if risk_str == "High risk":
        st.error(f"{icon} {risk_str} (threshold = {threshold:.2f})")
    else:
        st.success(f"{icon} {risk_str} (threshold = {threshold:.2f})")
    st.markdown("---")
    st.markdown("### 📊 About the Data and Model Behind This Risk Factor Prediction")
    st.markdown("""
The Breast Cancer Surveillance Consortium (BCSC) dataset contains millions of mammogram records, risk factors, and cancer outcomes from diverse populations in the U.S.  
[Learn more about BCSC](https://www.bcsc-research.org/)
""")
    # Figure 1: Age of participant by group
    st.image("figures/age.png", use_column_width=True)
    st.markdown("""
The majority of study participants fall in the 45–74 age range, with the highest counts in the 50–59 and 55–59 age groups.
""")

    # Figure 2: BMI by group
    st.image("figures/bmi.png", use_column_width=True)
    st.markdown("""
The largest number of participants, both with and without breast cancer history, are in the lower BMI groups (10–24.99 and 25–29.99).
""")

    # Figure 3: first degree cancer history
    st.image("figures/family_history.png", use_column_width=True)
    st.markdown("""
Most participants do not have a first-degree family history of breast cancer, regardless of their own cancer history. However, among those with a history of breast cancer (orange bars), a larger proportion report a family history of the disease compared to those without cancer.
""")

    # Figure 4: Feature Importance
    st.image("figures/feature_importance_xgb.png", use_column_width=True)
    st.markdown("""
**Which Factors Matter Most?**  
The feature importance plot shows which risk factors contribute most to the model's predictions.
""")

# --- Tab 2: Wellness & Tracker ---
with tab2:
    st.header("Glow and Grow")
    st.write("Here are some tips and a simple tracker to help you with meditation, diet, and exercise.")

    # Tips section
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

    # Tracker section
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

    # Additional resources
    st.subheader("Additional Resources")
    # YouTube video links
    st.markdown("**YouTube Videos:**")
    videos = {
        "Mindfulness Meditation for Cancer Support": "https://www.youtube.com/watch?v=1ZYbU82GVz4&t=31s",
        "Gentle Move for All": "https://www.youtube.com/watch?v=Ev6yE55kYGw&t=169s",
        "Healthy Eating During Cancer Treatment": "https://www.youtube.com/shorts/kkk8UPd7l38"
    }
    for title, url in videos.items():
        st.markdown(f"- [{title}]({url})")

    # Local support groups
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
