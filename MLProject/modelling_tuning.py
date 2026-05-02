import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset lokal
df = pd.read_csv("social_anxiety_preprocessed.csv")

X = df.drop(["Anxiety Level (1-10)", "Anxiety Category"], axis=1)
y = df["Anxiety Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Setup MLflow lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Social Anxiety Tuning")

# Model dan parameter tuning
model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

with mlflow.start_run(run_name="RandomForest_Tuning_ManualLogging"):
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Manual logging parameter
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("best_params", grid_search.best_params_)
    mlflow.log_param("cv", 3)
    mlflow.log_param("scoring", "accuracy")

    # Manual logging metric
    mlflow.log_metric("accuracy", accuracy)

    # Simpan classification report
    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # Simpan confusion matrix sebagai gambar
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    # Simpan model
    mlflow.sklearn.log_model(best_model, "model")

    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(report)