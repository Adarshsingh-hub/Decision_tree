import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("./telco_churn.csv")

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors= "coerce")
data = data.dropna()

features = ["tenure", "MonthlyCharges", "TotalCharges"]

X = data[features]

scaler =StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)

clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

print(data[["tenure","MonthlyCharges","TotalCharges","Cluster"]].head())

plt.scatter(data["tenure"], data["MonthlyCharges"], c=data["Cluster"])
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title("Customer Clusters")
plt.show()

print(data.groupby("Cluster")[features].mean())