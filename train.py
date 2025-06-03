import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer, fbeta_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle
import logging
import os

# -------------------------------
# Preprocessing for Modeling
# -------------------------------
def preprocess_data(data):
    """Preprocess data for modeling."""
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col])

    cols_to_drop = ['time', 'customer_id', 'email', 'informational', 'became_member_on']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    X = data.drop(columns=['offer_successful'])
    y = data['offer_successful'].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# -------------------------------
# Model Evaluation
# -------------------------------
def model_evaluation(model, X_test, y_test):
    """Evaluate model performance."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Accuracy: {acc:.6f}, F1-score: {f1:.6f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("Classification Report:")
    print(classification_report(y_test, preds))
    return acc, f1

# -------------------------------
# Train and Evaluate Models
# -------------------------------
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train, evaluate, and save multiple models."""
    models = {
        'AdaBoost': AdaBoostClassifier(DecisionTreeClassifier(random_state=141), random_state=141),
        'DecisionTree': DecisionTreeClassifier(random_state=141),
        'RandomForest': RandomForestClassifier(random_state=141),
        'LightGBM': LGBMClassifier(random_state=141),
        'CatBoost': CatBoostClassifier(random_state=141, verbose=0)
    }

    trained_models = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        print(f"=== Evaluating {name} ===")
        acc, f1 = model_evaluation(model, X_test, y_test)
        trained_models[name] = model

        # Save the model to a pickle file
        filename = f"{name.lower()}_model.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Saved {name} model to {filename}")

        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values(by='importance', ascending=False)
            print(f"\nFeature Importance ({name}):")
            print(fi.to_string(index=False))

    return trained_models

# -------------------------------
# Refine LightGBM
# -------------------------------
def refine_lightgbm(X_train, y_train, X_test, y_test):
    """Refine LightGBM model using RandomizedSearchCV and save the best model."""
    lgb = LGBMClassifier(random_state=141)
    scorer = make_scorer(fbeta_score, beta=0.5)

    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_depth': [3, 4, 5, 6, None],
        'min_child_samples': [10, 20, 30],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    lgb_random = RandomizedSearchCV(
        estimator=lgb,
        param_distributions=param_grid,
        scoring=scorer,
        n_iter=20,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=4
    )

    print("\nRefining LightGBM with RandomizedSearchCV...")
    lgb_random.fit(X_train, y_train)

    print("\nBest Parameters for LightGBM:")
    print(lgb_random.best_params_)
    print("\nEvaluating Refined LightGBM...")
    best_model = lgb_random.best_estimator_
    model_evaluation(best_model, X_test, y_test)

    # Save the refined LightGBM model to a pickle file
    filename = "refined_lightgbm_model.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(best_model, file)
    print(f"Saved refined LightGBM model to {filename}")

    return best_model

# -------------------------------
# Main
# -------------------------------
def main():
    """Execute model training pipeline using master_offer_analysis.csv and save models."""
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    print("Loading preprocessed data...")
    try:
        data = pd.read_csv('master_offer_analysis.csv')
    except FileNotFoundError:
        print("Error: 'master_offer_analysis.csv' not found. Please ensure the file exists.")
        return

    print("Preprocessing data for modeling...")
    X_train, X_test, y_train, y_test = preprocess_data(data)

    print("Training and evaluating models...")
    trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("Refining LightGBM model...")
    refined_model = refine_lightgbm(X_train, y_train, X_test, y_test)

    print("\nModel training pipeline complete. All models saved as pickle files.")

if __name__ == "__main__":
    main()