import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Starbucks Offer Predictor",
    page_icon="☕",
    layout="wide"
)

st.markdown("""
    <style>
    #MainMenu { visibility: hidden; }
    header:not(.hero-banner) { visibility: hidden; }

    ::-webkit-scrollbar { width: 0; height: 0; }
    ::-webkit-scrollbar-track { display: none; }
    ::-webkit-scrollbar-thumb { display: none; }
    html { scrollbar-width: none; -ms-overflow-style: none; }
    html, body, .block-container { margin: 0; padding: 0; overflow-x: hidden; }

    .main .block-container, .stApp > div {
        max-width: 775px;
        margin: auto;
        padding: 2rem 1.5rem;
        background-color: #121212;
        color: #e0e0e0;
    }

    .hero-banner {
        background: linear-gradient(135deg, rgba(13, 71, 161, 0.25) 0%, rgba(30, 136, 229, 0.25) 100%);
        padding: 3rem 1.5rem !important;
        text-align: center;
        margin: 0 0 2rem 0 !important;
        color: #e0e0e0;
        border-radius: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        border: 1px solid #4fc3f7;
        animation: fadeIn 1.5s ease-in;
    }
    .hero-banner h1 { font-size: clamp(1.5rem, 5vw, 1.75rem); font-weight: bold; }
    .hero-banner p { font-size: clamp(0.75rem, 2.5vw, 0.875rem); }

    .prediction-result {
        background: linear-gradient(to bottom, #00695c 0%, #004d40 80%);
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        width: 100% !important;
        max-width: 750px !important;
        color: #e0e0e0;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        border: 1px solid #4fc3f7;
    }

    .footer {
        text-align: center;
        margin: 4rem auto 1rem;
        padding: 2.5rem 1rem 1.5rem;
        background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(30, 39, 46, 0.2) 100%);
        border-radius: 12px;
        font-size: 15px;
        color: #d1d5db;
        max-width: 800px;
        position: relative;
        z-index: 100;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .footer p {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    .footer a {
        color: #60a5fa;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .footer a:hover {
        color: #3b82f6;
        text-decoration: none;
    }

    .final-decision {
        margin:0;
        padding: 1rem 2rem 2rem 2.5rem;
        border-radius: 10px;
        color: #ffffff;
    }
    .final-decision.success {
        background: linear-gradient(to bottom, rgba(46, 125, 50, .1), rgba(27, 94, 32, .1));
        border: 1px solid #66bb6a;
    }
    .final-decision.failure {
        background: linear-gradient(135deg, rgba(211, 47, 47, 0.1), rgba(183, 28, 28, 0.1));
        border: 1px solid #ef5350;
    }
    .final-decision h3 {
        margin: 0;
        color: #ffffff;
    }
    .final-decision p {
        margin: 0;
        font-size: 1.1rem;
        color: #ffffff;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

if 'auto_fill' not in st.session_state:
    st.session_state.auto_fill = None

auto_success = dict(
    age=35, income=75000, reward=10.0, difficulty=10, duration=7, total_amount=100.0,
    web=True, social=True, mobile=True, bogo=1, discount=0, gender=0,
    start_year=2017, start_month=6
)
auto_fail = dict(
    age=25, income=30000, reward=2.0, difficulty=20, duration=30, total_amount=10.0,
    web=False, social=False, mobile=False, bogo=0, discount=0, gender=1,
    start_year=2017, start_month=6
)

if st.session_state.auto_fill == 'yes':
    form_values = auto_success
elif st.session_state.auto_fill == 'no':
    form_values = auto_fail
else:
    form_values = dict(
        age=40, income=60000, reward=5.0, difficulty=10, duration=7, total_amount=50.0,
        web=False, social=False, mobile=False, bogo=0, discount=0, gender=0,
        start_year=2017, start_month=6
    )

st.markdown("""
    <header class="hero-banner">
        <h1>☕ Starbucks Rewards Response Predictor</h1>
        <p>Maximize the impact of your Starbucks Rewards offers by predicting which members will respond</p>
    </header>
""", unsafe_allow_html=True)

st.markdown("## Enter Starbucks Rewards Offer Details")
with st.form("offer_form"):
    col1, col2 = st.columns(2)
    @st.cache_data
    def load_features():
        df = pd.read_csv("master_offer_analysis.csv")
        drop = ['offer_successful', 'time', 'customer_id', 'email', 'informational', 'became_member_on']
        return [col for col in df.columns if col not in drop], df['offer_id'].unique()

    features, valid_offer_ids = load_features()
    with col1:
        age = st.slider("Age", 18, 100, form_values['age'])
        income = st.slider("Income ($)", 30000, 120000, form_values['income'], 1000)
        reward = st.slider("Reward", 0.0, 20.0, form_values['reward'], 0.5)
        difficulty = st.slider("Difficulty", 0, 20, form_values['difficulty'])
        duration = st.slider("Duration (days)", 1, 30, form_values['duration'])
        total_amount = st.slider("Total Amount Spent", 0.0, 1000.0, form_values['total_amount'], 10.0)
        start_year = st.selectbox("Start Year", [2013, 2014, 2015, 2016, 2017, 2018], index=[2013, 2014, 2015, 2016, 2017, 2018].index(form_values['start_year']))

    with col2:
        offer_id = st.selectbox("Offer ID", options=list(valid_offer_ids))
        offer_type = st.selectbox("Offer Type", ["bogo", "discount", "informational"])
        channel = st.multiselect("Channel(s)", ["web", "email", "mobile", "social"], default=["web"])
        gender = st.radio("Gender", [0, 1], index=form_values['gender'], format_func=lambda x: "Male" if x == 0 else "Female")
        web = st.checkbox("Web Channel", value=form_values['web'])
        social = st.checkbox("Social Channel", value=form_values['social'])
        mobile = st.checkbox("Mobile Channel", value=form_values['mobile'])
        bogo = st.selectbox("BOGO Offer", [0, 1], index=form_values['bogo'], format_func=lambda x: 'Yes' if x == 1 else 'No')
        discount = st.checkbox("Discount Offer", value=form_values['discount'])
        start_month = st.slider("Start Month", 1, 12, form_values['start_month'])

    colA, colB, colC = st.columns(3)
    with colA:
        if st.form_submit_button("🔁 Auto-fill Likely Success"):
            st.session_state.auto_fill = 'yes'
            st.rerun()
    with colB:
        if st.form_submit_button("🔁 Auto-fill Likely Fail"):
            st.session_state.auto_fill = 'no'
            st.rerun()
    with colC:
        submit = st.form_submit_button("🚀 Predict Offer Success")

# Prediction Logic
if submit:
    # Load models and data features
    @st.cache_resource
    def load_models():
        models = {}
        names = {
            'randomforest': 'Random Forest', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost',
            'adaboost': 'AdaBoost', 'decisiontree': 'Decision Tree', 'refined_lightgbm': 'Refined LightGBM'
        }
        model_dir = 'Models'
        if not os.path.exists(model_dir):
            st.error("Models directory not found.")
            return models, None
        for f in os.listdir(model_dir):
            if f.endswith('.pkl'):
                key = f.replace('_model.pkl', '')
                title = names.get(key, key.title())
                with open(os.path.join(model_dir, f), 'rb') as file:
                    models[title] = pickle.load(file)
        return models

    @st.cache_data
    def load_features():
        df = pd.read_csv("master_offer_analysis.csv")
        drop = ['offer_successful', 'time', 'customer_id', 'email', 'informational', 'became_member_on']
        return [col for col in df.columns if col not in drop], df['offer_id'].unique()

    features, valid_offer_ids = load_features()
    models = load_models()

    if offer_id not in valid_offer_ids:
        st.error("Invalid offer ID. Please select from known offer IDs in the dataset.")
    else:
        le = LabelEncoder()
        le.fit(valid_offer_ids)
        input_data = {
            'offer_id': le.transform([offer_id])[0],
            'total_amount': total_amount,
            'reward': reward,
            'difficulty': difficulty,
            'duration': duration,
            'web': int(web),
            'social': int(social),
            'mobile': int(mobile),
            'bogo': bogo,
            'discount': int(discount),
            'gender': gender,
            'age': age,
            'income': income,
            'start_year': start_year,
            'start_month': start_month
        }
        input_df = pd.DataFrame([input_data], columns=features)

        results = []
        for name, model in models.items():
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else 0.0
            outcome = "Yes" if pred == 1 else "No"
            results.append({'Model': name, 'Prediction': outcome, 'Success_Probability': proba})

        st.markdown("### 📊 Starbucks Member Response Predictions")
        pred_df = pd.DataFrame(results)
        fig = px.bar(pred_df, x='Model', y='Success_Probability',
                     color='Success_Probability', color_continuous_scale='Bluered',
                     text_auto='.1f', title='Response Rate by Model')
        fig.update_layout(yaxis_tickformat='.0%', yaxis_range=[0, 1], showlegend=False)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% Response Threshold", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)

        votes = sum(1 for r in results if r['Success_Probability'] > 0.5)
        is_success = votes > len(results) / 2
        decision = "✅  This Starbucks Rewards member is likely to respond to this offer." if is_success else "❌  This Starbucks Rewards member is unlikely to respond to this offer."

        decision_class = "success" if is_success else "failure"
        st.markdown(f"""
        <div class="final-decision {decision_class}">
            <h3>Final Decision</h3>
            <p>{decision}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <footer class="footer">
        <p>Crafted with ☕ & ❤️ by <a href="https://github.com/omkarbhad" target="_blank">Omkar Bhad</a></p>
    </footer>
""", unsafe_allow_html=True)
