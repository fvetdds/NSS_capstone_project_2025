import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# Page config
st.set_page_config(page_title="Breast Cancer Risk & Survival", layout="wide")
st.markdown("""
    <style>
    /* Make tab labels larger and yellow */
    .stTabs [data-baseweb="tab"] {
        font-size: 3.5rem;
        color: #008080;  /* teal */
        background: #FFD700;
        border-bottom: 2px solid transparent;
        transition: border-color 0.3s;
    }
    /* Highlight the active tab with bold, underline, white text, and yellow border */
    .stTabs [aria-selected="true"] {
        font-weight: bold;
        color: #fff !important;     /* white text */
        border-bottom: 4px solid #ffd700 !important;  /* thick gold underline */
        background: #232323;        /* slight dark bg */
    }
    /* Make the tab row more visible */
    .stTabs [role="tablist"] {
        border-bottom: 1px solid #ffd700;
        margin-bottom: 2em;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <h1 style='color:#008080; font-size:2.8rem; font-weight:bold; margin-bottom:0.5em;'>🎗️ EmpowerHER</h1>
""", unsafe_allow_html=True)



# Load models and data
BASE_DIR = Path(__file__).resolve().parent
# Classification model
model = joblib.load(BASE_DIR / "models" / "bcsc_xgb_model.pkl")
threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Risk Insights", "Thrive Prediction", "Mind & Move"])

# --- Tab 1: Breast Cancer Risk Predictor ---
with tab1:
    st.sidebar.header("Your information for risk prediction")
    def sel(label, opts):
        return st.sidebar.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])

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

    # Collect sidebar inputs
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
st.image("figures/age.png", width=450)
st.markdown("""
**BCSC Data Source**  
The Breast Cancer Surveillance Consortium (BCSC) dataset contains millions of mammogram records, risk factors, and cancer outcomes from diverse populations in the U.S.  
[Learn more about BCSC](https://www.bcsc-research.org/)
""")

# Figure 2: Feature Distribution Example
st.image("figures/feature_distribution.png", width=450)
st.markdown("""
**Key Features Used for Prediction**  
Features include age, race/ethnicity, age at first period, age at first birth, family history, breast density, hormone use, and more. The chart above shows the distribution of selected features in the dataset.
""")

# Figure 3: XGBoost Classifier
st.image("figures/xgboost_model.png", width=450)
st.markdown("""
**XGBoost Classifier**  
We use a gradient-boosted decision tree model (XGBoost) that combines many simple trees for accurate prediction of breast cancer risk.
""")

# Figure 4: Feature Importance
st.image("figures/feature_importance.png", width=450)
st.markdown("""
**Which Factors Matter Most?**  
The feature importance plot shows which risk factors contribute most to the model's predictions.
""")

# Figure 5: Risk Calculation Pipeline
st.image("figures/prediction_pipeline.png", width=450)
st.markdown("""
**How is Your Risk Calculated?**  
Your answers are transformed into numbers and passed into our trained XGBoost model, which outputs a probability. This is compared to a data-driven threshold to classify your risk as low or high.
""")


# --- Tab 2: 5-Year Survival Predictor ---
with tab2:
    st.header("5-Year Survival Probability by Gene Mutation Markers")
    
# --- Tab 3: Wellness & Tracker ---
with tab3:
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
        },
        {"name": "Susan G. Komen Nashville", "phone": "(615) 673-6633", "website": "https://komen.org/nashville"},
        {"name": "Vanderbilt Breast Cancer Support Group", "phone": "(615) 322-3900", "website": "https://www.vicc.org/support-groups"},
        {"name": "Alive Hospice Cancer Support", "phone": "(615) 327-1085", "website": "https://alivehospice.org"}
    ]
    for grp in support_groups:
        st.markdown(f"- **{grp['name']}**: {grp['phone']} | [Website]({grp['website']})")
