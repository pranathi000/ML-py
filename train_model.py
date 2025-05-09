"""
Autism Prediction Model Training Script

This script:
1. Loads and preprocesses the dataset
2. Trains multiple models (Decision Tree, Random Forest, XGBoost)
3. Selects the best model using cross-validation
4. Saves the model and encoders for use in the Streamlit app
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

def main(): 
    print("Starting model training...")
    
    # Check if the data file exists
    data_file = "train.csv"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please make sure it's in the current directory.")
        return
    
    # 1. Data Loading & Preprocessing
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_file)
    
    # Convert age column datatype to integer
    df["age"] = df["age"].astype(int)
    
    # Dropping ID & age_desc column
    df = df.drop(columns=["ID", "age_desc"])
    
    # Define the mapping dictionary for country names
    mapping = {
        "Viet Nam": "Vietnam",
        "AmericanSamoa": "United States",
        "Hong Kong": "China"
    }
    
    # Replace values in the country column
    df["contry_of_res"] = df["contry_of_res"].replace(mapping)
    
    # Handle missing values in ethnicity and relation column
    df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})
    df["relation"] = df["relation"].replace({
        "?": "Others",
        "Relative": "Others",
        "Parent": "Others",
        "Health care professional": "Others"
    })
    
    # 2. Label Encoding
    print("Applying label encoding...")
    # Identify columns with "object" data type
    object_columns = df.select_dtypes(include=["object"]).columns
    
    # Initialize a dictionary to store the encoders
    encoders = {}
    
    # Apply label encoding and store the encoders
    for column in object_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        encoders[column] = label_encoder  # Saving the encoder for this column
    
    # Save the encoders as a pickle file
    with open("encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print("Encoders saved to 'encoders.pkl'")
    
    # 3. Handle outliers
    print("Handling outliers...")
    # Function to replace the outliers with median
    def replace_outliers_with_median(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        median = df[column].median()
        
        # Replace outliers with median value
        df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
        
        return df
    
    # Replace outliers in the "age" column
    df = replace_outliers_with_median(df, "age")
    
    # Replace outliers in the "result" column
    df = replace_outliers_with_median(df, "result")
    
    # 4. Train-Test Split
    print("Splitting data into train and test sets...")
    X = df.drop(columns=["Class/ASD"])
    y = df["Class/ASD"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Apply SMOTE for handling class imbalance
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # 6. Model Training and Hyperparameter Tuning
    print("Training models and tuning hyperparameters...")
    
    # Initialize models
    decision_tree = DecisionTreeClassifier(random_state=42)
    random_forest = RandomForestClassifier(random_state=42)
    xgboost_classifier = XGBClassifier(random_state=42)
    
    # Hyperparameter grids for RandomizedSearchCV
    param_grid_dt = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30, 50, 70],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    
    param_grid_rf = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }
    
    param_grid_xgb = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.5, 0.7, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0]
    }
    
    # Perform RandomizedSearchCV for each model
    print("Tuning Decision Tree...")
    random_search_dt = RandomizedSearchCV(
        estimator=decision_tree,
        param_distributions=param_grid_dt,
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )
    random_search_dt.fit(X_train_smote, y_train_smote)
    
    print("Tuning Random Forest...")
    random_search_rf = RandomizedSearchCV(
        estimator=random_forest,
        param_distributions=param_grid_rf,
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )
    random_search_rf.fit(X_train_smote, y_train_smote)
    
    print("Tuning XGBoost...")
    random_search_xgb = RandomizedSearchCV(
        estimator=xgboost_classifier,
        param_distributions=param_grid_xgb,
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )
    random_search_xgb.fit(X_train_smote, y_train_smote)
    
    # Get the model with best score
    print("Selecting best model...")
    best_model = None
    best_score = 0
    best_model_name = ""
    
    if random_search_dt.best_score_ > best_score:
        best_model = random_search_dt.best_estimator_
        best_score = random_search_dt.best_score_
        best_model_name = "Decision Tree"
    
    if random_search_rf.best_score_ > best_score:
        best_model = random_search_rf.best_estimator_
        best_score = random_search_rf.best_score_
        best_model_name = "Random Forest"
    
    if random_search_xgb.best_score_ > best_score:
        best_model = random_search_xgb.best_estimator_
        best_score = random_search_xgb.best_score_
        best_model_name = "XGBoost"
    
    print(f"Best Model: {best_model_name}")
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")
    
    # Save the best model
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("Best model saved to 'best_model.pkl'")
    
    # 7. Evaluation on test data
    print("Evaluating model on test data...")
    y_test_pred = best_model.predict(X_test)
    
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()