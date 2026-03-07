"""
Main execution file for AI Intrusion Detection System
"""

from src.data_loader import prepare_datasets
from src.model import train_model, save_model
from src.evaluate import evaluate_model
from src.explain import explain_model


def main():

    print("Loading and preparing dataset...")

    X_train, X_test, y_train, y_test, feature_names = prepare_datasets(
        "data/UNSW_NB15_training-set.csv",
        "data/UNSW_NB15_testing-set.csv"
    )

    print("\nTraining AI intrusion detection model...")

    model = train_model(X_train, y_train)

    print("\nEvaluating model performance...")

    best_threshold, best_accuracy = evaluate_model(
        model,
        X_test,
        y_test,
        feature_names
    )

    print("\nGenerating model explanations...")

    explain_model(model, X_test, feature_names)

    print("\nSaving trained model...")

    save_model(model)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()