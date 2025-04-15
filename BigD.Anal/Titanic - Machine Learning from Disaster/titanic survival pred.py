import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# Model selection
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
# Evaluation
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Classification models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

file_path = r'C:\Users\99kos\igloo\BigD.Anal\Titanic - Machine Learning from Disaster'

train_df = pd.read_csv(file_path+r'\train.csv')
test_df = pd.read_csv(file_path+r'\test.csv')

def handle_missing_values(df):

    most_common_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(most_common_embarked)

    df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

    df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

    df['Cabin'] = df['Cabin'].fillna('Unknown')

    return df

train_df = handle_missing_values(train_df)
test_df = handle_missing_values(test_df)

def feature_engineering_and_encoding(df, is_train=True):
    X = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])  
    
    if is_train:
        y = df['Survived']
        X = X.drop(columns=['Survived'])
    else:
        y = None

    categorical_cols = ['Sex', 'Embarked']
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())  # Standardizing numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OneHotEncoding categorical features
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    X_processed = preprocessor.fit_transform(X)

    numerical_cols_transformed = numerical_cols
    categorical_cols_transformed = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)

    all_columns = numerical_cols_transformed + list(categorical_cols_transformed)

    X_processed_df = pd.DataFrame(X_processed, columns=all_columns)

    return X_processed_df, y

X_train, y_train = feature_engineering_and_encoding(train_df, is_train=True)
X_test, _ = feature_engineering_and_encoding(test_df, is_train=False)

X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

param_grids = {
    'Logistic Regression': {'C': [0.1, 0.5, 1, 10]},
    'Random Forest': {'n_estimators': [100, 200, 300],
                      'max_depth': [5, 10, 15],
                      'min_samples_split': [2, 5, 10]},
    'Gradient Boosting': {'n_estimators': [100, 200, 300],
                          'learning_rate': [0.01, 0.05, 0.1],
                          'max_depth': [3, 5, 7]},
    'Support Vector Classifier': {'C': [0.1, 1, 10],
                                  'gamma': ['scale', 'auto'],
                                  'kernel': ['rbf', 'linear']},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 10],
                            'weights': ['uniform', 'distance'],
                            'metric': ['minkowski', 'euclidean']},
    'XGBoost': {'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5]}
}

def create_model(model_name):
    if model_name == 'Logistic Regression':
        return LogisticRegression(random_state=1, solver='liblinear')
    elif model_name == 'Random Forest':
        return RandomForestClassifier(random_state=1)
    elif model_name == 'Gradient Boosting':
        return GradientBoostingClassifier(random_state=1)
    elif model_name == 'Support Vector Classifier':
        return SVC(random_state=1, probability=True)
    elif model_name == 'K-Nearest Neighbors':
        return KNeighborsClassifier()
    elif model_name == 'XGBoost':
        return xgb.XGBClassifier(random_state=1, eval_metric='logloss')

def tune_hyperparameters(model_name, model, param_grid, X_train, y_train):
    print(f"\n🔧 Tuning hyperparameters for {model_name}...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5, n_jobs=-1,
        scoring='accuracy',
        error_score='raise'
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_


def train_and_evaluate_model_with_cv(model, X_train, y_train, cv_folds=10):
    """
    Trains the given model using cross-validation and evaluates it on the training data.
    
    Parameters:
    - model: The classification model to train and evaluate.
    - X_train: Training features.
    - y_train: Training labels.
    - cv_folds: Number of folds for cross-validation (default is 10).
    
    Prints:
    - Accuracy
    - F1 Score
    - Cross-Validation Mean Accuracy
    - Confusion Matrix
    """
    
    # cross_validate
    cv_results = cross_validate(model, X_train, y_train, cv=cv_folds, scoring='accuracy', return_train_score=False)
    cv_mean_accuracy = cv_results['test_score'].mean()
    cv_std_accuracy = cv_results['test_score'].std()

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_train)
    
    # Evaluate
    accuracy = accuracy_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred, average='weighted')

    # Print the results
    print(f"\n🔹 {model.__class__.__name__} Model Evaluation:")
    print(f"📌 Accuracy: {accuracy * 100:.2f}%")
    print(f"📌 F1 Score: {f1 * 100:.2f}%")
    print(f"📌 Cross-Validation Accuracy (Mean ± Std): {cv_mean_accuracy * 100:.2f}% ± {cv_std_accuracy * 100:.2f}%")
    
    # Display confusion matrix
    print("\n🔍 Confusion Matrix:")
    print(confusion_matrix(y_train, y_pred))
    print("=======================================")

# Loop through all models and apply hyperparameter tuning and evaluation
for model_name in param_grids.keys():
    model = create_model(model_name)
    best_model = tune_hyperparameters(model_name, model, param_grids[model_name], X_train, y_train)
    train_and_evaluate_model_with_cv(best_model, X_train, y_train)