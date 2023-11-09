import pandas as pd
from sklearn.impute import SimpleImputer

from data_preparation import load_data, clean_data, preprocess_data

def create_features(data):
    """
    Create additional features in the dataset, such as 'FamilySize' and handle missing values using mean imputation.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
        pd.DataFrame: The DataFrame with additional features and missing values handled.
    """
    data['FamilySize'] = data['Siblings/Spouses Aboard'] + data['Parents/Children Aboard']
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data
