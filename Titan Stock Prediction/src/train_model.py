import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("data/processed/titan_features.csv")

feature_cols = [
    "Daily_Return",
    "MA5",
    "MA5_to_MA20",
    "Price_to_MA20",
    "Lag1_Return",
    "Lag2_Return",
    "Volatility_5d",
    "Volume_Change"
]

X = df[feature_cols]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"Logistic Regression Test Accuracy: {lr_accuracy:.2%}")

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print(f"Decision Tree Test Accuracy: {dt_accuracy:.2%}")

dt_train_accuracy = accuracy_score(
    y_train,
    dt_model.predict(X_train)
)

print(f"Decision Tree Training Accuracy: {dt_train_accuracy:.2%}")

depths = range(1, 16)

train_accuracies = []
test_accuracies = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracies.append(
        accuracy_score(y_train, train_pred)
    )

    test_accuracies.append(
        accuracy_score(y_test, test_pred)
    )

plt.figure(figsize=(8,5))

plt.plot(depths, train_accuracies, marker='o', label='Training Accuracy')
plt.plot(depths, test_accuracies, marker='o', label='Testing Accuracy')

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Underfitting vs Overfitting")

plt.legend()
plt.grid(True)

plt.savefig("plots/underfitting_vs_overfitting.png", dpi=300)

plt.show()

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Random Forest Test Accuracy: {rf_accuracy:.2%}")
rf_train_accuracy = accuracy_score(
    y_train,
    rf_model.predict(X_train)
)

print(f"Random Forest Training Accuracy: {rf_train_accuracy:.2%}")

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)

print(f"XGBoost Test Accuracy: {xgb_accuracy:.2%}")
xgb_train_accuracy = accuracy_score(
    y_train,
    xgb_model.predict(X_train)
)

print(f"XGBoost Training Accuracy: {xgb_train_accuracy:.2%}")
xgb_cm = confusion_matrix(y_test, xgb_predictions)

xgb_cm = confusion_matrix(y_test, xgb_predictions)

plt.figure(figsize=(6, 5))

plt.imshow(xgb_cm, interpolation='nearest', cmap='Blues')
plt.title("XGBoost Test Confusion Matrix")
plt.colorbar()

classes = ["Down", "Up"]
tick_marks = range(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, xgb_cm[i, j],
                 ha="center",
                 va="center",
                 color="black",
                 fontsize=12)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig("plots/xgb_confusion_matrix.png", dpi=300)

plt.show()

print("\n=== XGBoost Classification Report ===\n")

print(
    classification_report(
        y_test,
        xgb_predictions,
        target_names=["DOWN (0)", "UP (1)"]
    )
)

models = [
    "Logistic\nRegression",
    "Decision\nTree",
    "Random\nForest",
    "XGBoost"
]

accuracies = [
    lr_accuracy * 100,
    dt_accuracy * 100,
    rf_accuracy * 100,
    xgb_accuracy * 100
]

plt.figure(figsize=(8,5))

bars = plt.bar(models, accuracies)

plt.ylabel("Test Accuracy (%)")
plt.title("Comparison of Machine Learning Models")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.2,
        f"{height:.2f}%",
        ha='center'
    )

plt.ylim(0, 100)

plt.savefig("plots/model_comparison.png", dpi=300)

plt.show()

importance = xgb_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

plt.figure(figsize=(8,5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xticks(rotation=45, ha="right")

plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig("plots/xgb_feature_importance.png", dpi=300)

plt.show()

feature_importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

print("Feature importance saved successfully.")

report = classification_report(
    y_test,
    xgb_predictions,
    target_names=["DOWN (0)", "UP (1)"]
)

print(report)

with open(
    "results/classification_report.txt",
    "w"
) as file:
    file.write(report)

print("Classification report saved successfully.")

summary = f"""
Titan Stock Price Prediction Project

Best Model: XGBoost

Training Accuracy : {xgb_train_accuracy:.2%}
Testing Accuracy  : {xgb_accuracy:.2%}

Most Important Feature:
{feature_importance.iloc[0]['Feature']}

Least Important Feature:
{feature_importance.iloc[-1]['Feature']}
"""

with open(
    "results/project_summary.txt",
    "w"
) as file:
    file.write(summary)

print("Project summary saved successfully.")