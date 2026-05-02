import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("social_anxiety_preprocessed.csv")

X = df.drop(["Anxiety Level (1-10)", "Anxiety Category"], axis=1)
y = df["Anxiety Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.sklearn.autolog()

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

mlflow.log_metric("accuracy_manual", accuracy)

print("Accuracy:", accuracy)
