import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sqlite3
import hashlib
import os
from PIL import Image

# Database setup for user authentication
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                 (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('diabetes.csv')  # Update path if needed
        # Handle missing values (replace 0s with median for relevant columns)
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            data[col] = data[col].replace(0, data[col].median())
        return data
    except FileNotFoundError:
        st.error("diabetes.csv not found. Please ensure the dataset is in the same directory.")
        return None

def preprocess_data(data):
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, y, scaler

# Train models with improvements for class imbalance
@st.cache_resource
def train_models(X, y):
    # Use stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    # Logistic Regression with class weighting
    lr = LogisticRegression(random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)

    # XGBoost with GridSearch and scale_pos_weight
    xgb = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300]
    }
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Evaluation metrics
    metrics = {}
    for model, name in [(lr, 'Logistic Regression'), (grid_search.best_estimator_, 'XGBoost')]:
        y_pred = model.predict(X_test)
        metrics[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        }

    return lr, grid_search.best_estimator_, X_train, X_test, y_train, y_test, metrics

# New real-world function: Provide diabetes risk factor insights
def get_risk_insights(prediction):
    if prediction == 1:
        return """
        **High Risk Insights:**
        - Consult a healthcare professional immediately for further testing.
        - Maintain a balanced diet low in sugar and refined carbs.
        - Engage in regular physical activity (at least 150 minutes/week).
        - Monitor blood glucose levels regularly.
        - Consider weight management if BMI is high.
        """
    else:
        return """
        **Low Risk Insights:**
        - Continue healthy lifestyle habits to prevent onset.
        - Regular check-ups are recommended, especially if family history exists.
        - Stay active and maintain a healthy weight.
        - Eat a diet rich in fruits, vegetables, and whole grains.
        """

# New real-world function: Calculate BMI category
def calculate_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Streamlit app
def main():
    # Initialize database
    init_db()

    # Custom CSS for styling (enhanced with more colors)
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #34495e;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .input-box {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-box {
        background-color: #ecf0f1;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .positive-result {
        color: red;
        font-weight: bold;
    }
    .negative-result {
        color: green;
        font-weight: bold;
    }
    .insights-box {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)

    # App title and logo
    st.markdown('<div class="title">Early Diabetes Prediction System</div>', unsafe_allow_html=True)
    # Assuming logo.png exists in the same directory
    if os.path.exists('logo.png'):
        logo = Image.open('logo.png')
        st.image(logo, width=150, caption="Diabetes Prediction App")
    else:
        st.warning("Logo file (logo.png) not found. Place it in the same directory.")

    # Session state for authentication
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    # Login/Signup interface
    if not st.session_state.logged_in:
        st.sidebar.header("Authentication")
        auth_choice = st.sidebar.radio("Choose Action", ["Login", "Signup"])

        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if auth_choice == "Login":
            if st.sidebar.button("Login"):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")
        else:
            if st.sidebar.button("Signup"):
                if register_user(username, password):
                    st.sidebar.success("Registered successfully! Please login.")
                else:
                    st.sidebar.error("Username already exists")
    else:
        # Main app content
        st.sidebar.write(f"Welcome, {st.session_state.username}!")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

        # Load and preprocess data
        data = load_data()
        if data is None:
            return
        X, y, scaler = preprocess_data(data)
        lr, xgb, X_train, X_test, y_train, y_test, metrics = train_models(X, y)

        # Sidebar for model selection and navigation
        st.sidebar.header("Options")
        model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "XGBoost"])
        page = st.sidebar.radio("Navigate", ["Prediction", "Model Evaluation", "EDA"])

        if page == "Prediction":
            # Input form for patient data
            st.markdown('<div class="input-box">', unsafe_allow_html=True)
            st.subheader("Enter Patient Data")
            cols = st.columns(2)
            inputs = {}
            features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

            for i, feature in enumerate(features):
                with cols[i % 2]:
                    inputs[feature] = st.number_input(feature, min_value=0.0, step=0.1, 
                                                    help=f"Enter value for {feature}")

            if st.button("Predict Diabetes Risk"):
                # Validate inputs
                if all(inputs[feat] >= 0 for feat in features):
                    # Prepare input data
                    input_data = np.array([[inputs[feat] for feat in features]])
                    input_scaled = scaler.transform(input_data)

                    # Make prediction
                    model = lr if model_choice == "Logistic Regression" else xgb
                    pred = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0][1]

                    # Display results with colors
                    st.markdown("### Prediction Results")
                    result = "Positive (High Risk)" if pred == 1 else "Negative (Low Risk)"
                    color_class = "positive-result" if pred == 1 else "negative-result"
                    st.markdown(f"**Diabetes Prediction**: <span class='{color_class}'>{result}</span>", unsafe_allow_html=True)
                    st.write(f"**Confidence Score**: {prob:.2%}")

                    # New: BMI category
                    bmi_category = calculate_bmi_category(inputs['BMI'])
                    st.markdown(f'<div class="warning-box">**BMI Category**: {bmi_category}</div>', unsafe_allow_html=True)

                    # New: Risk insights
                    insights = get_risk_insights(pred)
                    st.markdown('<div class="insights-box">', unsafe_allow_html=True)
                    st.subheader("Real-World Insights")
                    st.markdown(insights)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # ROC Curve
                    st.subheader("ROC Curve")
                    y_pred_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})', 
                            color='#3498db')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend()
                    st.pyplot(fig)

                    # Feature importance (for XGBoost)
                    if model_choice == "XGBoost":
                        st.subheader("Feature Importance (SHAP)")
                        explainer = shap.TreeExplainer(xgb)
                        shap_values = explainer.shap_values(X_test)
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
                        st.pyplot(fig)
                else:
                    st.error("Please ensure all input values are non-negative.")
            st.markdown('</div>', unsafe_allow_html=True)

        elif page == "Model Evaluation":
            st.subheader("Model Evaluation Metrics")
            st.write("Performance metrics for trained models:")
            for model_name, metric_values in metrics.items():
                st.markdown(f"#### {model_name}")
                cols = st.columns(4)
                for i, (metric, value) in enumerate(metric_values.items()):
                    with cols[i]:
                        st.markdown(f'<div class="metric-box"><b>{metric}</b><br>{value:.4f}</div>', 
                                   unsafe_allow_html=True)

        else:  # EDA
            st.subheader("Exploratory Data Analysis")
            st.write("Visualizing correlations between features and diabetes outcome.")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            # Feature distributions
            st.subheader("Feature Distributions")
            feature = st.selectbox("Select Feature for Distribution", data.columns[:-1])
            fig, ax = plt.subplots()
            sns.histplot(data[feature], kde=True, color='#3498db')
            ax.set_title(f'Distribution of {feature}')
            st.pyplot(fig)

if __name__ == "__main__":
    main()