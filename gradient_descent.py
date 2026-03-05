import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("./telco_churn.csv")

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

features = ["tenure", "MonthlyCharges", "TotalCharges"]

X = data[features]
y = data["Churn"].map({"Yes":1, "No": 0})

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 3
)

model.fit(X_train, y_train)

preds = model.predict(X_val)

acc = accuracy_score(y_val, preds)

print(acc)