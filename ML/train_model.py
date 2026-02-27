import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load dataset
data = pd.read_csv("mindguard_dataset.csv")

X = data.drop("risk", axis=1)
y = data["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning (basic but solid)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    class_weight={0:1, 1:1.2, 2:1.5},
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", round(accuracy, 3))

# Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("\nCross Validation Accuracy:", round(cv_scores.mean(), 3))

# Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Feature Importance (Sorted)
importances = model.feature_importances_
feature_importance = sorted(
    zip(X.columns, importances),
    key=lambda x: x[1],
    reverse=True
)

print("\nFeature Importance:\n")
for name, score in feature_importance:
    print(f"{name}: {round(score, 3)}")

# Save model
joblib.dump(model, "risk_model.pkl")
print("\nModel saved as risk_model.pkl")