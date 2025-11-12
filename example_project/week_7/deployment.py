"""
Deploy the trained student dropout prediction model using Streamlit.

Run with: streamlit run deployment.py

This script creates an interactive web application where users can input student information
and receive predictions about their graduation status (Graduate, Dropout, or Enrolled).

The deployment process involves:
1. Loading the saved model that was trained in final_model.py (run this first)
2. Loading the dataset to recreate the preprocessing pipeline (scaler)
3. Creating a user-friendly interface to collect student information
4. Applying the same preprocessing steps used during training (feature engineering + scaling)
5. Making predictions and displaying results in an easy-to-understand format

Resources:
- Streamlit documentation: https://docs.streamlit.io/
- Streamlit widgets: https://docs.streamlit.io/library/api-reference/widgets
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from exploratory_data_analysis import feature_engineering
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


@st.cache_data
def load_model_and_scaler() -> tuple:
    """
    Load the trained model and create a scaler fitted on training data.

    This function is decorated with @st.cache_data, which means Streamlit will cache
    the results. This is important because:
    - Loading the model and dataset is expensive (takes time)
    - We don't want to reload them every time the user interacts with the app
    - The cache ensures we only load once, then reuse the loaded objects

    Returns:
        tuple: A tuple containing (model, scaler, X_train_engineered)
            - model: The trained Random Forest classifier
            - scaler: A StandardScaler fitted on the training data
            - X_train_engineered: The engineered training features (used to ensure
              the scaler and model expect the same feature order)

    Resources:
    - Streamlit caching: https://docs.streamlit.io/library/api-reference/performance/st.cache_data
    - joblib.load: https://joblib.readthedocs.io/en/latest/generated/joblib.load.html
    """
    # Get the path to the model file
    # We use Path(__file__).parent to get the directory containing this script
    # This ensures the path works regardless of where the script is run from
    script_dir = Path(__file__).parent
    model_path = script_dir / "models" / "best_random_forest_model.joblib"

    # Check if the model file exists
    if not model_path.exists():
        st.error(
            f"Model file not found at {model_path}. "
            "Please make sure you've run final_model.py to train and save the model first."
        )
        st.stop()  # Stop execution if model is missing

    # Load the trained model from the saved file
    model = joblib.load(model_path)
    st.success(f"Model loaded successfully from {model_path}")

    # Load the dataset to recreate the preprocessing pipeline
    dataset = fetch_ucirepo(id=697)

    # Separate features and target
    X = dataset.data.features.copy()
    y = dataset.data.targets["Target"]

    # Split the data the same way we did during training
    # We use the same random_state (42) to ensure we get the same split
    # This is critical: the scaler must be fitted on the same training data
    # that was used during model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Apply feature engineering to the training data
    # This creates the same engineered features that were used during training
    X_train_engineered = feature_engineering(X_train.copy())

    # Create and fit the scaler on the training data
    # We fit on training data only, then use the same transformation for new predictions
    scaler = StandardScaler()
    scaler.fit(X_train_engineered)

    return model, scaler, X_train_engineered


def preprocess_user_input(user_data: dict) -> NDArray:
    """
    Preprocess user input to match the format expected by the model.

    This function:
    1. Converts the user input dictionary to a pandas DataFrame
    2. Applies the same feature engineering used during training
    3. Scales the features using the fitted scaler
    4. Returns the preprocessed data ready for prediction

    Args:
        user_data: Dictionary containing user input values for all features

    Returns:
        NDArray: Preprocessed feature array ready for model prediction

    Note:
        The order of features in the DataFrame must match the order used during training.
        This is why we use X_train_engineered.columns to ensure consistency.
    """
    # Convert user input dictionary to a pandas DataFrame
    # We create a DataFrame with a single row (one student's data)
    user_df = pd.DataFrame([user_data])

    # Apply feature engineering to create the same derived features used during training
    user_df_engineered = feature_engineering(user_df.copy())

    # Ensure the columns are in the same order as the training data
    # This is critical: the model expects features in a specific order
    # We reorder the columns to match the training data
    model, scaler, X_train_engineered = load_model_and_scaler()
    user_df_engineered = user_df_engineered[X_train_engineered.columns]

    # Scale the features using the same scaler that was fitted on training data
    user_scaled = scaler.transform(user_df_engineered)

    return user_scaled


# ============================================================================
# STREAMLIT APP
# ============================================================================


def main() -> None:
    """
    Main function that creates and runs the Streamlit web application.

    This function:
    1. Sets up the page configuration and title
    2. Loads the model and scaler (cached for performance)
    3. Creates input forms for collecting student information
    4. Makes predictions when the user clicks the predict button
    5. Displays results in a user-friendly format
    """

    # Configure the Streamlit page
    # These settings control how the page appears in the browser
    st.set_page_config(
        page_title="Student Dropout Prediction",
        page_icon="🎓",
        layout="wide",
    )

    # Display the main title and description
    st.title("🎓 Student Graduation Status Predictor")
    st.markdown(
        """
        This application predicts a student's graduation status based on their academic
        and personal information. The model was trained on historical student data and
        can predict whether a student will **Graduate**, **Dropout**, or remain **Enrolled**.
        
        Fill in the information below and click "Predict" to see the prediction!
        """
    )

    # Load the model and scaler
    model, scaler, X_train_engineered = load_model_and_scaler()

    # ============================================================================
    # CREATE INPUT FORMS
    # ============================================================================
    # Create a form to collect all user inputs
    # Using st.form prevents the page from rerunning on every input change
    with st.form("student_info_form"):
        st.header("Student Information")

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Information")
            # Marital Status: 1-6 (categorical)
            marital_status = st.selectbox(
                "Marital Status",
                options=[1, 2, 3, 4, 5, 6],
                help="Student's marital status (1-6)",
            )

            # Gender: 0 or 1 (binary)
            gender = st.selectbox(
                "Gender",
                options=[0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
                help="Student's gender (0=Female, 1=Male)",
            )

            # Age at enrollment: numeric
            age = st.number_input(
                "Age at Enrollment",
                min_value=17,
                max_value=70,
                value=20,
                help="Student's age when they enrolled",
            )

            # International: 0 or 1 (binary)
            international = st.selectbox(
                "International Student",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether the student is international (0=No, 1=Yes)",
            )

            # Displaced: 0 or 1 (binary)
            displaced = st.selectbox(
                "Displaced",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether the student is displaced (0=No, 1=Yes)",
            )

            # Educational special needs: 0 or 1 (binary)
            special_needs = st.selectbox(
                "Educational Special Needs",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether the student has educational special needs (0=No, 1=Yes)",
            )

        with col2:
            st.subheader("Application Information")
            # Application mode: 1-57 (categorical)
            application_mode = st.number_input(
                "Application Mode",
                min_value=1,
                max_value=57,
                value=1,
                help="Mode of application (1-57)",
            )

            # Application order: 0-9 (numeric)
            application_order = st.number_input(
                "Application Order",
                min_value=0,
                max_value=9,
                value=0,
                help="Order of application (0-9)",
            )

            # Course: 33-9991 (categorical, but we'll use numeric input)
            course = st.number_input(
                "Course",
                min_value=33,
                max_value=9991,
                value=171,
                help="Course code (33-9991)",
            )

            # Daytime/evening attendance: 0 or 1 (binary)
            attendance = st.selectbox(
                "Daytime/Evening Attendance",
                options=[0, 1],
                format_func=lambda x: "Evening" if x == 0 else "Daytime",
                help="Attendance schedule (0=Evening, 1=Daytime)",
            )

            # Previous qualification: 1-43 (categorical)
            prev_qualification = st.number_input(
                "Previous Qualification",
                min_value=1,
                max_value=43,
                value=1,
                help="Previous qualification code (1-43)",
            )

            # Previous qualification grade: 95.0-190.0 (numeric)
            prev_qualification_grade = st.number_input(
                "Previous Qualification Grade",
                min_value=95.0,
                max_value=190.0,
                value=120.0,
                help="Grade from previous qualification (95.0-190.0)",
            )

            # Admission grade: 95.0-190.0 (numeric)
            admission_grade = st.number_input(
                "Admission Grade",
                min_value=95.0,
                max_value=190.0,
                value=120.0,
                help="Admission grade (95.0-190.0)",
            )

        # Family Information Section
        st.subheader("Family Information")
        fam_col1, fam_col2 = st.columns(2)

        with fam_col1:
            # Nacionality: 1-109 (categorical)
            nationality = st.number_input(
                "Nationality",
                min_value=1,
                max_value=109,
                value=1,
                help="Student's nationality code (1-109)",
            )

            # Mother's qualification: 1-44 (categorical)
            mother_qualification = st.number_input(
                "Mother's Qualification",
                min_value=1,
                max_value=44,
                value=19,
                help="Mother's qualification code (1-44)",
            )

            # Father's qualification: 1-44 (categorical)
            father_qualification = st.number_input(
                "Father's Qualification",
                min_value=1,
                max_value=44,
                value=19,
                help="Father's qualification code (1-44)",
            )

        with fam_col2:
            # Mother's occupation: 0-194 (categorical)
            mother_occupation = st.number_input(
                "Mother's Occupation",
                min_value=0,
                max_value=194,
                value=0,
                help="Mother's occupation code (0-194)",
            )

            # Father's occupation: 0-195 (categorical)
            father_occupation = st.number_input(
                "Father's Occupation",
                min_value=0,
                max_value=195,
                value=0,
                help="Father's occupation code (0-195)",
            )

        # Financial Information Section
        st.subheader("Financial Information")
        fin_col1, fin_col2, fin_col3 = st.columns(3)

        with fin_col1:
            # Debtor: 0 or 1 (binary)
            debtor = st.selectbox(
                "Debtor",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether the student is a debtor (0=No, 1=Yes)",
            )

        with fin_col2:
            # Tuition fees up to date: 0 or 1 (binary)
            tuition_up_to_date = st.selectbox(
                "Tuition Fees Up to Date",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether tuition fees are up to date (0=No, 1=Yes)",
            )

        with fin_col3:
            # Scholarship holder: 0 or 1 (binary)
            scholarship_holder = st.selectbox(
                "Scholarship Holder",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether the student is a scholarship holder (0=No, 1=Yes)",
            )

        # Academic Performance - First Semester
        st.subheader("Academic Performance - First Semester")
        sem1_col1, sem1_col2, sem1_col3 = st.columns(3)

        with sem1_col1:
            curricular_units_1st_sem_credited = st.number_input(
                "Curricular Units 1st Sem (Credited)",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of credited curricular units in 1st semester",
            )

            curricular_units_1st_sem_enrolled = st.number_input(
                "Curricular Units 1st Sem (Enrolled)",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of enrolled curricular units in 1st semester",
            )

            curricular_units_1st_sem_evaluations = st.number_input(
                "Curricular Units 1st Sem (Evaluations)",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of evaluated curricular units in 1st semester",
            )

        with sem1_col2:
            curricular_units_1st_sem_approved = st.number_input(
                "Curricular Units 1st Sem (Approved)",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of approved curricular units in 1st semester",
            )

            curricular_units_1st_sem_grade = st.number_input(
                "Curricular Units 1st Sem (Grade)",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                help="Average grade for 1st semester curricular units (0.0-20.0)",
            )

            curricular_units_1st_sem_without_evaluations = st.number_input(
                "Curricular Units 1st Sem (Without Evaluations)",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of curricular units without evaluations in 1st semester",
            )

        # Academic Performance - Second Semester
        st.subheader("Academic Performance - Second Semester")
        sem2_col1, sem2_col2, sem2_col3 = st.columns(3)

        with sem2_col1:
            curricular_units_2nd_sem_credited = st.number_input(
                "Curricular Units 2nd Sem (Credited)",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of credited curricular units in 2nd semester",
            )

            curricular_units_2nd_sem_enrolled = st.number_input(
                "Curricular Units 2nd Sem (Enrolled)",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of enrolled curricular units in 2nd semester",
            )

            curricular_units_2nd_sem_evaluations = st.number_input(
                "Curricular Units 2nd Sem (Evaluations)",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of evaluated curricular units in 2nd semester",
            )

        with sem2_col2:
            curricular_units_2nd_sem_approved = st.number_input(
                "Curricular Units 2nd Sem (Approved)",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of approved curricular units in 2nd semester",
            )

            curricular_units_2nd_sem_grade = st.number_input(
                "Curricular Units 2nd Sem (Grade)",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                help="Average grade for 2nd semester curricular units (0.0-20.0)",
            )

            curricular_units_2nd_sem_without_evaluations = st.number_input(
                "Curricular Units 2nd Sem (Without Evaluations)",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of curricular units without evaluations in 2nd semester",
            )

        # Economic Indicators
        st.subheader("Economic Indicators")
        econ_col1, econ_col2, econ_col3 = st.columns(3)

        with econ_col1:
            unemployment_rate = st.number_input(
                "Unemployment Rate",
                min_value=-10.0,
                max_value=20.0,
                value=0.0,
                help="Unemployment rate at time of enrollment",
            )

        with econ_col2:
            inflation_rate = st.number_input(
                "Inflation Rate",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                help="Inflation rate at time of enrollment",
            )

        with econ_col3:
            gdp = st.number_input(
                "GDP",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                help="GDP growth rate at time of enrollment",
            )

        # Submit button
        # When clicked, this will process the form and make a prediction
        submitted = st.form_submit_button(
            "🔮 Predict Graduation Status", use_container_width=True
        )

    # ============================================================================
    # MAKE PREDICTION
    # ============================================================================
    if submitted:
        # Create a dictionary with all the user inputs
        # The keys must match the column names in the original dataset
        user_data = {
            "Marital Status": marital_status,
            "Application mode": application_mode,
            "Application order": application_order,
            "Course": course,
            "Daytime/evening attendance": attendance,
            "Previous qualification": prev_qualification,
            "Previous qualification (grade)": prev_qualification_grade,
            "Nacionality": nationality,
            "Mother's qualification": mother_qualification,
            "Father's qualification": father_qualification,
            "Mother's occupation": mother_occupation,
            "Father's occupation": father_occupation,
            "Admission grade": admission_grade,
            "Displaced": displaced,
            "Educational special needs": special_needs,
            "Debtor": debtor,
            "Tuition fees up to date": tuition_up_to_date,
            "Gender": gender,
            "Scholarship holder": scholarship_holder,
            "Age at enrollment": age,
            "International": international,
            "Curricular units 1st sem (credited)": curricular_units_1st_sem_credited,
            "Curricular units 1st sem (enrolled)": curricular_units_1st_sem_enrolled,
            "Curricular units 1st sem (evaluations)": curricular_units_1st_sem_evaluations,
            "Curricular units 1st sem (approved)": curricular_units_1st_sem_approved,
            "Curricular units 1st sem (grade)": curricular_units_1st_sem_grade,
            "Curricular units 1st sem (without evaluations)": curricular_units_1st_sem_without_evaluations,
            "Curricular units 2nd sem (credited)": curricular_units_2nd_sem_credited,
            "Curricular units 2nd sem (enrolled)": curricular_units_2nd_sem_enrolled,
            "Curricular units 2nd sem (evaluations)": curricular_units_2nd_sem_evaluations,
            "Curricular units 2nd sem (approved)": curricular_units_2nd_sem_approved,
            "Curricular units 2nd sem (grade)": curricular_units_2nd_sem_grade,
            "Curricular units 2nd sem (without evaluations)": curricular_units_2nd_sem_without_evaluations,
            "Unemployment rate": unemployment_rate,
            "Inflation rate": inflation_rate,
            "GDP": gdp,
        }

        try:
            user_scaled = preprocess_user_input(user_data)

            # Make prediction using the loaded model
            prediction = model.predict(user_scaled)[0]

            # Get prediction probabilities for each class
            probabilities = model.predict_proba(user_scaled)[0]

            # Get the class names (the order matches the probabilities)
            class_names = model.classes_

            # Create a dictionary mapping class names to probabilities
            prob_dict = dict(zip(class_names, probabilities))

            # ============================================================================
            # DISPLAY RESULTS
            # ============================================================================
            st.success("✅ Prediction Complete!")

            # Display the main prediction
            st.header("Prediction Result")
            # Use different colors and emojis based on the prediction
            if prediction == "Graduate":
                st.markdown(
                    f'<div style="background-color:#d4edda;padding:20px;border-radius:10px;text-align:center">'
                    f'<h1 style="color:#155724;">🎓 {prediction}</h1>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif prediction == "Dropout":
                st.markdown(
                    f'<div style="background-color:#f8d7da;padding:20px;border-radius:10px;text-align:center">'
                    f'<h1 style="color:#721c24;">⚠️ {prediction}</h1>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:  # Enrolled
                st.markdown(
                    f'<div style="background-color:#fff3cd;padding:20px;border-radius:10px;text-align:center">'
                    f'<h1 style="color:#856404;">📚 {prediction}</h1>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.subheader("Prediction Confidence")
            st.markdown(
                "The model's confidence (probability) for each possible outcome:"
            )

            # Create a bar chart of probabilities
            prob_df = pd.DataFrame(
                {
                    "Outcome": list(prob_dict.keys()),
                    "Probability": list(prob_dict.values()),
                }
            )
            prob_df = prob_df.sort_values("Probability", ascending=False)

            # Display as a bar chart
            st.bar_chart(prob_df.set_index("Outcome"))

            # Display probabilities as percentages
            for outcome, prob in sorted(
                prob_dict.items(), key=lambda x: x[1], reverse=True
            ):
                # Create a progress bar for each outcome
                st.write(f"**{outcome}**: {prob:.1%}")
                st.progress(prob)

        except Exception as e:
            # If something goes wrong, display an error message
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info(
                "Please check that all input values are within the expected ranges "
                "and try again."
            )

    # ============================================================================
    # ADDITIONAL INFORMATION
    # ============================================================================
    # Add an expander with additional information about the model
    with st.expander("ℹ️ About This Model"):
        st.markdown(
            """
            **Model Details:**
            - **Algorithm**: Random Forest Classifier
            - **Training Dataset**: UCI Student Dropout Dataset (ID: 697)
            - **Target Classes**: Graduate, Dropout, Enrolled
            - **Features**: 36 original features + 4 engineered features
            
            **How It Works:**
            1. Your input is processed using the same feature engineering steps used during training
            2. Features are scaled to match the training data distribution
            3. The trained Random Forest model makes a prediction
            4. Probabilities are calculated for each possible outcome
            
            **Note**: This is a predictive model based on historical data. 
            Predictions are not guarantees and should be used as one tool among many 
            for understanding student outcomes.
            
            **Resources:**
            - Dataset: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
            - Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            """
        )


if __name__ == "__main__":
    main()
