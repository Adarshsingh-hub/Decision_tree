import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv("./telco_churn.csv")
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors = 'coerce')
data = data.dropna()

features = ['tenure', 'MonthlyCharges', 'TotalCharges']

X= data[features]
y = data["Churn"].map({"Yes":1 , "No":0})

X_train, X_val, y_train, y_val = train_test_split(
    X,y , test_size=0.3, random_state=42, stratify=y
)

tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    random_state=42
)

tree.fit(X_train, y_train)

for depth in [2, 3, 4, 5, 6, 8, 10]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    
    print (f"Depth={depth} → Accuracy={acc:.4f}")
    
plt.figure(figsize=(12,8))
plot_tree(tree, feature_names=features, class_names=["No", "Yes"], filled=True)
plt.show()