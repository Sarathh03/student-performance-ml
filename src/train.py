# src/train.py

import pandas as pd
import joblib
import os
os.makedirs("model", exist_ok=True)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r'C:\Users\Sarat\Downloads\student-performance-ml\data\StudentsPerformance.csv')

# Create target variable: Pass / Fail
df['pass_fail'] = df['math score'].apply(lambda x: 1 if x >= 40 else 0)

# Separate features and target
X = df.drop(['pass_fail'], axis=1)
y = df['pass_fail']

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include='object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_model = None
best_accuracy = 0

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc}")
    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

import os
os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/encoders.pkl")


