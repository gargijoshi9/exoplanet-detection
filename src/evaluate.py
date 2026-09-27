"""
Model evaluation module for Exoplanet Detection.
"""

from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, output_dir="results"):
    """
    Evaluate the classifier and save performance metrics and confusion matrix.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm


if __name__ == "__main__":
    pass
