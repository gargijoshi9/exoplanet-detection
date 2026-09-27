"""
Model training module for Exoplanet Detection.
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, model_save_path="models/model.joblib"):
    """
    Train classification model and save it to models directory.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    return model


if __name__ == "__main__":
    pass
