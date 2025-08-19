# Early Diabetes Prediction System

This is a Streamlit-based web application for predicting diabetes risk using machine learning models (Logistic Regression and XGBoost). The app includes user authentication, exploratory data analysis (EDA), model evaluation, and real-world health insights based on predictions.

# Features





User Authentication: Secure login and signup system using SQLite for user data storage.



Prediction: Predict diabetes risk based on patient data with Logistic Regression or XGBoost models.



Model Evaluation: Display performance metrics (Accuracy, Precision, Recall, ROC-AUC) for trained models.



Exploratory Data Analysis (EDA): Visualize feature correlations and distributions.



Real-World Insights: Provide actionable health advice based on prediction results.



BMI Category: Calculate and display BMI category for user input.



SHAP Analysis: Feature importance visualization for XGBoost model.



ROC Curve: Display ROC curve for model performance.

# Prerequisites





Python 3.8+



Git



Streamlit Cloud account (for deployment)



GitHub account

# Installation





Clone the repository:

git clone https://github.com/your-username/early-diabetes-prediction.git
cd early-diabetes-prediction



Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



# Install dependencies:

pip install -r requirements.txt



Download the dataset (diabetes.csv) from Kaggle and place it in the project root directory.



(Optional) Place a logo.png file in the project root for the app logo.

Running Locally

Run the Streamlit app:

streamlit run app.py

The app will be available at http://localhost:8501.

# Deployment

Deploy to Streamlit Cloud





Push the project to a GitHub repository (see Git Commands below).



Log in to Streamlit Cloud.



Create a new app, select the GitHub repository, and specify the main script (app.py).



Ensure requirements.txt is in the repository root.



Deploy the app. Streamlit Cloud will handle dependency installation and hosting.

# Deploy to GitHub

Use the following Git commands to initialize, commit, and push the project to GitHub:

# Initialize a new Git repository
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit of Early Diabetes Prediction System"

# Add remote repository (replace with your GitHub repo URL)
git remote add origin https://github.com/your-username/early-diabetes-prediction.git

# Push to GitHub
git push -u origin main

Dataset

The app uses the PIMA Indians Diabetes Dataset (diabetes.csv), which must be placed in the project root directory. The dataset includes features like Pregnancies, Glucose, BloodPressure, etc., and the target variable Outcome (0 or 1).

File Structure

early-diabetes-prediction/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
├── README.md          # Project documentation
├── diabetes.csv       # Dataset (not included in repo, download separately)
├── logo.png           # Optional logo file
├── users.db           # SQLite database for user authentication (auto-generated)

Notes





The app assumes diabetes.csv is in the project root. If not found, it displays an error.



The users.db file is automatically created for user authentication.



The logo.png file is optional; the app shows a warning if not found.



Models are trained with class imbalance handling (stratified split and scale_pos_weight for XGBoost).

License

This project is licensed under the MIT License.