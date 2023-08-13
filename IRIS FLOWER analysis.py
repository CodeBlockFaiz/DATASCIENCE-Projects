import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


df = sns.load_dataset("iris")


X = df.drop("species", axis=1)
y = df["species"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)


y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


target_names = df["species"].unique()
print(classification_report(y_test, y_pred, target_names=target_names))
