import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preparation import preprocess_data, clean_data, load_data
import joblib
import mlflow
import numpy as np
import shap
import matplotlib.pyplot as plt

def train_model(X_train, X_test, y_train, y_test):
    """
    Train a RandomForestClassifier on preprocessed data and evaluate its performance.

    Args:
        X_train (pd.DataFrame): Features for training.
        X_test (pd.DataFrame): Features for testing.
        y_train (pd.Series): Target labels for training.
        y_test (pd.Series): Target labels for testing.
    """
    # Prompt user for parameter values
    n_estimators = int(input("Enter the n-estimators parameter: "))
    random_state = int(input("Enter random_state parameter: "))
    # Initialize the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance (you can use other metrics as well)
    accuracy = accuracy_score(y_test, y_pred)




    # Log parameters and metrics in MLflow
    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": n_estimators,
            "random_state": random_state,
        })
        mlflow.log_metrics({
            "accuracy": accuracy,
        })

    print(f"Accuracy: {accuracy}")

    return model

def explain_model_predictions(model, X, data_point, class_index=1):
    """
    Explain model predictions using SHAP values for a specific data point.

    Args:
        model: Trained model to explain.
        data: The dataset used for explaining predictions.
        index: The index of the data point to explain.

    Returns:
        None
    """
    # Convert the data point and features to NumPy array
    data_point_array = np.array(data_point)
    X_array = X

    # Select only the numerical features from the data point (assuming they are contiguous)
    numerical_data_point = data_point_array[:len(X_array[0])]



    # Create a TreeExplainer for the model
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the chosen data point
    """shap_values = explainer.shap_values(numerical_data_point)

    # Visualize the SHAP values for the chosen data point
    shap.initjs()
    shap.force_plot(explainer.expected_value[class_index], shap_values[class_index], data_point)


    # Visualize explanations for all data points in the dataset
    shap.summary_plot(shap_values, X, class_names=[f"Class {i}" for i in range(len(shap_values))], plot_type ="bar")
"""

if __name__ == "__main__":
    data_file = 'data/titanic.csv'
    data = load_data(data_file)
    cleaned_data = clean_data(data)

    # Load preprocessed data from data_preparation script
    X_train, X_test, y_train, y_test = preprocess_data(cleaned_data)


    # Train and evaluate the RandomForestClassifier
    trained_model = train_model(X_train, X_test, y_train, y_test)
    model_file = 'trained_model.pkl'

    # Choose the data point to explain (replace with your desired feature vector)
    data_point_to_explain = X_test[0]

    # Explain the model's prediction for the chosen data point
    explain_model_predictions(trained_model, X_train, data_point_to_explain)
    joblib.dump(trained_model, model_file)