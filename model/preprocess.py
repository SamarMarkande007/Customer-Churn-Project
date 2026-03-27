
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def get_preprocessor():
    num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    cat_cols = ['Geography', 'Gender']
    bin_cols = ['Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols),
        ('bin', 'passthrough', bin_cols)
    ])

    return preprocessor

