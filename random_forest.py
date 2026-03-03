import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

data = pd.read_csv("./telco_churn.csv")

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors = 'coerce')
data = data.dropna()

features = ["tenure", "MonthlyCharges", "TotalCharges"]

X = data[features]
y = data["Churn"].map({"Yes": 1, "No": 0})

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state=42, stratify = y)


best_acc = 0
best_params = None

for n in [100, 200]:
    for depth in [None, 5, 10]:
        for min_split in [2, 5]:
            
            model = RandomForestClassifier(
                n_estimators=n,
                max_depth=depth,
                min_samples_split=min_split,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            
            print(f"Trees={n}, Depth={depth}, MinSplit={min_split} → Acc={acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_params = (n, depth, min_split)

print("\nBest Accuracy:", best_acc)
print("Best Params:", best_params)

#best accuracy at (100, 5, 5)