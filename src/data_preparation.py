import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(data_file):
    """
    Load data from a CSV file.

    Parameters:
        data_file (str): The path to the CSV file containing the data.

    Returns:
        pd.DataFrame: A DataFrame containing the data.

    """
    data = pd.read_csv(data_file)
    return data

def clean_data(data):
    """
    Clean the data by replacing missing values in the 'Age' column with the mean.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.

    """
    imputer = SimpleImputer(strategy='mean')
    data['Age'] = imputer.fit_transform(data['Age'].values.reshape(-1, 1))
    return data

def preprocess_data(data):
    """
    Preprocess the data by performing several steps, including splitting into training and test sets, converting categorical variables to binary variables, and scaling the features.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data to be preprocessed.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.

    """
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
    X = X.drop('Name', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
