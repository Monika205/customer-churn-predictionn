
import streamlit as st
import requests
import pandas as pd
import json
import os
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telecom Churn Prediction Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 20px;
}

.section-title {
    font-size: 24px;
    font-weight: 650;
    margin-top: 10px;
}

.metric-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background-color: #ffffff;
    text-align: center;
}

.risk-high {
    padding: 20px;
    border-radius: 12px;
    background-color: #fee2e2;
    border: 1px solid #fecaca;
}

.risk-low {
    padding: 20px;
    border-radius: 12px;
    background-color: #dcfce7;
    border: 1px solid #bbf7d0;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000"

# =========================================================
# LOAD MODEL INFORMATION
# =========================================================

model_name = "Logistic Regression"

try:
    model = joblib.load("final_model.pkl")
    model_name = type(model).__name__
except:
    pass

# =========================================================
# LOAD METRICS
# =========================================================

metrics_data = []

if os.path.exists("model_metrics.json"):
    try:
        with open("model_metrics.json", "r") as f:
            metrics_data = json.load(f)
    except:
        metrics_data = []

metrics_df = pd.DataFrame(metrics_data)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📡 Churn Platform")

    st.caption("AI-powered Telecom Customer Analytics")

    st.divider()

    st.markdown("### ⚙️ System Controls")

    st.text_input(
        "FastAPI Base URL",
        value=API_URL,
        disabled=True
    )

    # API STATUS
    try:
        health_response = requests.get(
            f"{API_URL}/health",
            timeout=2
        )

        if health_response.status_code == 200:
            st.success("🟢 API Online")
        else:
            st.warning("🟡 API Unstable")

    except:
        st.error("🔴 API Offline")

    st.divider()

    # MODEL INFORMATION
    st.markdown("### 🤖 Production Model")

    st.write("**Model:**")
    st.info(model_name)

    st.write("**Task:**")
    st.write("Binary Customer Churn Classification")

    st.write("**Framework:**")
    st.write("Scikit-learn")

    st.divider()

    # QUICK PRESETS
    st.markdown("### ⚡ Quick Presets")

    preset = st.selectbox(
        "Load Customer Profile",
        [
            "Custom Customer",
            "Low Risk Customer",
            "Moderate Risk Customer",
            "High Risk Customer"
        ]
    )

    st.divider()

    st.caption("Customer Churn Prediction System")
    st.caption("Production-ready ML prototype")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📡 Telecom Customer Churn Prediction & Retention System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-time customer risk scoring, churn probability assessment, '
    'model benchmarking and retention insights.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🎯 Single Customer Risk Assessment",
    "📦 Batch Customer Prediction",
    "📊 Model Benchmark & Metrics"
])

# =========================================================
# TAB 1 — SINGLE CUSTOMER
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">Customer Profile & Subscription Parameters</div>',
        unsafe_allow_html=True
    )

    st.write("Enter customer information to generate an individual churn risk score.")

    # -----------------------------------------------------
    # DEFAULT VALUES
    # -----------------------------------------------------

    if preset == "Low Risk Customer":
        default_tenure = 48
        default_contract = "Two year"
        default_internet = "DSL"
        default_monthly = 60.0
        default_total = 2880.0

    elif preset == "Moderate Risk Customer":
        default_tenure = 18
        default_contract = "One year"
        default_internet = "Fiber optic"
        default_monthly = 75.0
        default_total = 1350.0

    elif preset == "High Risk Customer":
        default_tenure = 2
        default_contract = "Month-to-month"
        default_internet = "Fiber optic"
        default_monthly = 90.0
        default_total = 180.0

    else:
        default_tenure = 1
        default_contract = "Month-to-month"
        default_internet = "DSL"
        default_monthly = 50.0
        default_total = 50.0

    # -----------------------------------------------------
    # CUSTOMER PROFILE
    # -----------------------------------------------------

    st.markdown("### 👤 1. Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        SeniorCitizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

    with col2:
        Partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        Dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col3:
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=default_tenure
        )

    # -----------------------------------------------------
    # SERVICES
    # -----------------------------------------------------

    st.markdown("### 🌐 2. Subscribed Services")

    col1, col2, col3 = st.columns(3)

    with col1:
        PhoneService = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        MultipleLines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        InternetService = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            index=[
                "DSL",
                "Fiber optic",
                "No"
            ].index(default_internet)
        )

    with col2:
        OnlineSecurity = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        OnlineBackup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        DeviceProtection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        TechSupport = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        StreamingTV = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        StreamingMovies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    # -----------------------------------------------------
    # BILLING
    # -----------------------------------------------------

    st.markdown("### 💳 3. Account & Billing")

    col1, col2, col3 = st.columns(3)

    with col1:
        Contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            index=[
                "Month-to-month",
                "One year",
                "Two year"
            ].index(default_contract)
        )

    with col2:
        PaperlessBilling = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col3:
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:
        MonthlyCharges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=default_monthly
        )

    with col2:
        TotalCharges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=default_total
        )

    st.divider()

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    predict_button = st.button(
        "🔮 Generate Churn Risk Assessment",
        use_container_width=True,
        type="primary"
    )

    if predict_button:

        customer_data = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }

        try:

            response = requests.post(
                f"{API_URL}/predict",
                json=customer_data,
                timeout=10
            )

            if response.status_code == 200:

                result = response.json()

                prediction = result["prediction"]
                probability = float(result["probability"])

                st.markdown("## 📌 Risk Assessment")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Prediction",
                        prediction
                    )

                with col2:
                    st.metric(
                        "Churn Probability",
                        f"{probability * 100:.2f}%"
                    )

                with col3:

                    if probability >= 0.70:
                        risk = "High Risk"
                    elif probability >= 0.40:
                        risk = "Moderate Risk"
                    else:
                        risk = "Low Risk"

                    st.metric(
                        "Risk Level",
                        risk
                    )

                # RISK MESSAGE

                if probability >= 0.70:

                    st.error(
                        "🚨 High churn risk detected. "
                        "Customer retention action is recommended."
                    )

                elif probability >= 0.40:

                    st.warning(
                        "⚠️ Moderate churn risk detected. "
                        "Customer should be monitored."
                    )

                else:

                    st.success(
                        "✅ Low churn risk detected. "
                        "Customer currently appears stable."
                    )

                # RETENTION RECOMMENDATIONS

                st.markdown("### 💡 Retention Recommendations")

                recommendations = []

                if Contract == "Month-to-month":
                    recommendations.append(
                        "Offer a longer-term contract with an attractive incentive."
                    )

                if tenure <= 12:
                    recommendations.append(
                        "Provide an onboarding or loyalty offer for the new customer."
                    )

                if MonthlyCharges >= 80:
                    recommendations.append(
                        "Review pricing and offer a suitable service bundle."
                    )

                if OnlineSecurity == "No":
                    recommendations.append(
                        "Consider offering an online security add-on."
                    )

                if TechSupport == "No":
                    recommendations.append(
                        "Consider offering technical support or assistance."
                    )

                if not recommendations:
                    recommendations.append(
                        "Continue regular engagement and loyalty monitoring."
                    )

                for recommendation in recommendations:
                    st.write("•", recommendation)

            else:

                st.error(
                    f"API Error ({response.status_code}): "
                    f"{response.text}"
                )

        except requests.exceptions.RequestException as e:

            st.error(
                f"❌ Could not connect to FastAPI: {e}"
            )

# =========================================================
# TAB 2 — BATCH PREDICTION
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">📦 Batch Customer Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a CSV file containing customer records. "
        "The system will generate churn predictions for each customer."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(uploaded_file)

            st.success(
                f"File loaded successfully — {len(batch_df)} customers found."
            )

            st.dataframe(
                batch_df.head(10),
                use_container_width=True
            )

            if st.button(
                "🚀 Run Batch Prediction",
                use_container_width=True
            ):

                predictions = []
                probabilities = []

                progress = st.progress(0)

                for i, row in batch_df.iterrows():

                    customer = row.to_dict()

                    # Remove ID if present
                    customer.pop("customerID", None)

                    try:

                        response = requests.post(
                            f"{API_URL}/predict",
                            json=customer,
                            timeout=10
                        )

                        if response.status_code == 200:

                            result = response.json()

                            predictions.append(
                                result["prediction"]
                            )

                            probabilities.append(
                                result["probability"]
                            )

                        else:

                            predictions.append("Error")
                            probabilities.append(None)

                    except:

                        predictions.append("Error")
                        probabilities.append(None)

                    progress.progress(
                        int(((i + 1) / len(batch_df)) * 100)
                    )

                batch_df["Prediction"] = predictions
                batch_df["Churn Probability"] = probabilities

                st.success("Batch prediction completed.")

                st.dataframe(
                    batch_df,
                    use_container_width=True
                )

                csv_data = batch_df.to_csv(index=False)

                st.download_button(
                    "⬇️ Download Prediction Results",
                    data=csv_data,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"Error reading CSV: {e}")

# =========================================================
# TAB 3 — MODEL BENCHMARK
# =========================================================

with tab3:

    st.markdown(
        '<div class="section-title">📊 Model Benchmark & Metrics</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Performance comparison of the machine learning models evaluated "
        "during the project."
    )

    if not metrics_df.empty:

        # Metric columns
        numeric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ]

        # Convert to percentage
        display_df = metrics_df.copy()

        for column in numeric_columns:
            if column in display_df.columns:
                display_df[column] = (
                    display_df[column] * 100
                ).round(2)

        st.dataframe(
            display_df,
            use_container_width=True
        )

        st.markdown("### 🏆 Model Performance")

        best_accuracy = metrics_df["Accuracy"].max()
        best_f1 = metrics_df["F1-Score"].max()
        best_recall = metrics_df["Recall"].max()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Best Accuracy",
                f"{best_accuracy * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Best F1-Score",
                f"{best_f1 * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Best Recall",
                f"{best_recall * 100:.2f}%"
            )

        # Performance chart
        st.markdown("### 📈 Performance Comparison")

        chart_df = metrics_df.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1-Score"]
        ]

        st.bar_chart(chart_df)

    else:

        st.info(
            "Model metrics file is not available yet. "
            "Run the model comparison step first."
        )

    st.markdown("### 🤖 Final Model")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Selected Model:** {model_name}")

    with col2:
        st.info("**Problem Type:** Binary Classification")

    st.markdown("### 🏗️ System Architecture")

    st.code(
        """
User
  ↓
Streamlit Web Interface
  ↓
FastAPI Prediction API
  ↓
Input Validation
  ↓
Preprocessing
  ↓
Logistic Regression Model
  ↓
Churn Prediction + Probability
  ↓
Risk Assessment / Retention Recommendation
        """,
        language="text"
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Telecom Customer Churn Prediction & Retention System | "
    "Machine Learning Deployment Project"
)
