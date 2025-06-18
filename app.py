import streamlit as st
import pandas as pd
from pickle import load
from streamlit_option_menu import option_menu
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "# This is a fraud detection app using SVM!"
    }
)

# --- CUSTOM CSS FOR STYLING ---
def load_css():
    st.markdown("""
        <style>
        /* --- General Text Color for Dark Theme --- */
        body {
            color: #FAFAFA;
        }
        
        /* --- About Page Cards --- */
        .about-card {
            background-color: #1A1A2E;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.3);
            border: 1px solid #FF4B4B;
        }
        .about-card h3 {
            color: #FF4B4B;
            margin-bottom: 1rem;
        }
        .about-card p, .about-card li {
            color: #E0E0E0;
            font-size: 1rem;
        }
        /* Style for links within the developer card for better visibility */
        .about-card a {
            color: #1E90FF; /* A nice, bright blue for links */
            text-decoration: none;
            font-weight: bold;
        }
        .about-card a:hover {
            text-decoration: underline;
            color: #46aeff;
        }

        /* --- Result Containers --- */
        .result-container {
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin-top: 2rem;
            box-shadow: 0 4px 12px 0 rgba(0,0,0,0.4);
        }
        .result-container.legit {
            background-color: rgba(40, 167, 69, 0.1);
            border: 1px solid #28a745;
        }
        .result-container.legit .result-text {
            color: #28a745;
        }
        .result-container.fraud {
            background-color: rgba(220, 53, 69, 0.1);
            border: 1px solid #dc3545;
        }
        .result-container.fraud .result-text {
            color: #dc3545;
        }
        .result-text {
            font-size: 1.75rem;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# --- DATA LOADING AND PREDICTION CLASS ---
class PickleData:
    def load_object(self, filename):
        with open(filename, 'rb') as f:
            return load(f)

    def predict_with_loaded_model(self, model_path, scaler_path, sample_data):
        model = self.load_object(model_path)
        scaler = self.load_object(scaler_path)
        sample_scaled = scaler.transform(sample_data)
        prediction = model.predict(sample_scaled)
        return prediction

# --- INITIALIZATION ---
pickler = PickleData()
feature_cols = ['TransactionAmount', 'CustomerAge', 'AccountBalance', 'ChannelEncoded', 'LoginAttempts']
channel_mapping = {"Online": 0, "Branch": 1, "ATM": 2}


# --- HORIZONTAL NAVIGATION BAR ---
page = option_menu(
    menu_title=None,
    options=["🔍 Prediction", "ℹ About"],
    icons=["search", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0E1117"},
        "icon": {"color": "#FAFAFA", "font-size": "18px"},
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px",
            "color": "#CCCCCC",
            "--hover-color": "#1A1A2E",
        },
        "nav-link-selected": {"background-color": "#FF4B4B", "color": "#FFFFFF"},
    },
)

# --- PAGE CONTENT ---
if page == "🔍 Prediction":
    st.title("🛡 Fraud Detection Using SVM")
    st.markdown("Enter transaction details below to get a real-time fraud prediction.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            transaction_amount = st.number_input("💰 Transaction Amount (PKR)", min_value=1.0, step=100.0, format="%.2f")
            customer_age = st.number_input("👤 Customer Age", min_value=1, max_value=100, step=1)
            account_balance = st.number_input("🏦 Account Balance (PKR)", min_value=0.0, step=500.0, format="%.2f")

        with col2:
            channel = st.selectbox("📡 Transaction Channel", options=list(channel_mapping.keys()))
            login_attempts = st.number_input("🔐 Login Attempts (Last 24h)", min_value=0, step=1)
            st.markdown("<br>", unsafe_allow_html=True) 

        submitted = st.form_submit_button("🧠 Predict Fraud Risk")

    if submitted:
        with st.spinner('Analyzing transaction...'):
            time.sleep(1) 
            
            input_data = {
                'TransactionAmount': [transaction_amount],
                'CustomerAge': [customer_age],
                'AccountBalance': [account_balance],
                'ChannelEncoded': [channel_mapping[channel]],
                'LoginAttempts': [login_attempts]
            }
            input_df = pd.DataFrame(input_data)[feature_cols]

            try:
                prediction = pickler.predict_with_loaded_model(
                    model_path="svm_model.pkl",
                    scaler_path="scaler.pkl",
                    sample_data=input_df
                )
                
                if prediction[0] == 1:
                    st.markdown(
                        """<div class="result-container fraud">
                            <p style="font-size: 5rem; margin-bottom: 0;">🚨</p>
                            <p class="result-text">High Risk: Fraud Detected!</p>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(
                        """<div class="result-container legit">
                            <p style="font-size: 5rem; margin-bottom: 0;">✅</p>
                            <p class="result-text">Low Risk: Transaction is Legitimate</p>
                        </div>""", unsafe_allow_html=True)
            except FileNotFoundError:
                st.error("Error: Model or scaler file not found. Please ensure 'svm_model.pkl' and 'scaler.pkl' are in the correct directory.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

elif page == "ℹ About":
    st.title("ℹ About This Application")
    st.markdown("---")

    st.markdown("""
    <div class="about-card">
        <h3>🛡 Intelligent Fraud Detection System</h3>
        <p>This web application uses a <strong>Support Vector Machine (SVM)</strong> model trained on financial transaction data to predict whether a transaction is <strong>fraudulent</strong> or <strong>legitimate</strong> in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="about-card">
            <h3>⚙ How it Works</h3>
            <ol>
                <li>User enters transaction details like amount, age, etc.</li>
                <li>Data is preprocessed with a standard scaler.</li>
                <li>The SVM model evaluates the risk of fraud based on the input.</li>
                <li>The result is displayed instantly: <strong>High Risk</strong> or <strong>Low Risk</strong>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card">
            <h3>👨‍💻 Tech Stack</h3>
            <ul>
                <li><strong>Python 🐍:</strong> Core programming language</li>
                <li><strong>Streamlit 🎈:</strong> For the interactive web interface</li>
                <li><strong>Scikit-learn 🔬:</strong> For the SVM model</li>
                <li><strong>Pandas 🐼:</strong> For data manipulation</li>
                <li><strong>Pickle 🧪:</strong> For model serialization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    st.warning("⚠ *Disclaimer:* This is a demo tool for educational purposes. Real-world fraud detection systems are significantly more complex.", icon="❗")