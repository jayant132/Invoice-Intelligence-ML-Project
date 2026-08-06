import os
import joblib

from data_preprocessing import (
    load_data,
    preprocess_data,
    prepare_data,
    split_and_scale,
)

from modeling_evaluation import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    tune_random_forest,
    evaluate_model,
)


def main():

    db_path = "../data/inventory.db"

    # Load data
    df = load_data(db_path)

    # Preprocess and create target
    df = preprocess_data(df)

    # Prepare features and target
    X, y = prepare_data(df)

    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Train models
    logistic_model = train_logistic_regression(X_train, y_train)

    decision_tree = train_decision_tree(X_train, y_train)

    random_forest = train_random_forest(X_train, y_train)

    tuned_random_forest = tune_random_forest(X_train, y_train)

    # Save the final model
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        tuned_random_forest,
        "models/random_forest.pkl"
    )

    print("Random Forest model saved successfully.")

    # Evaluate models
    evaluate_model(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression",
    )

    evaluate_model(
        decision_tree,
        X_test,
        y_test,
        "Decision Tree",
    )

    evaluate_model(
        random_forest,
        X_test,
        y_test,
        "Random Forest",
    )

    evaluate_model(
        tuned_random_forest,
        X_test,
        y_test,
        "Tuned Random Forest",
    )


if __name__ == "__main__":
    main()