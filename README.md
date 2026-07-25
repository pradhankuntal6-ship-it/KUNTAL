# KUNTAL
# Cardiovascular Disease Prediction
# A SIMPLE MACHINE LEARNING MODEL TO PREDICT HEART DISEASE
# CODE
import pandas as pd
df= pd.read_csv('D:\FOR PYTHON\CARDIO_TRAIN.csv', delimiter=';')
print(df)
print("\nMising values:\n",df.isnull().sum())
df=df.dropna()
x=df.drop("cardio",axis=1)
y=df["cardio"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaaler = StandardScaler()
x_scaled = scaaler.fit_transform(x)
x_tarin,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2, random_state=42)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,8))
df['cardio'].value_counts().plot(kind='bar', color=['green','red'])
plt.title("eart Diseases Count")
plt.xlabel("Cardio(0=Healthy, 1= disease)")
plt.ylabel('Count')
plt.show()
df.hist(figsize=(14,10))
plt.tight_layout()
plt.show()
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models={
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier()
}

accuracy_results = {}
for model_name, model in models.items():
  model.fit(x_tarin,y_train)
  y_pred = model.predict(x_test)
  acc = accuracy_score(y_test,y_pred)
  accuracy_results[model_name] = acc
  print(f"{model_name} Accuracy: {acc}")
  print(classification_report(y_test,y_pred))


plt.figure(figsize=(10,6))
plt.bar(accuracy_results.keys(),accuracy_results.values(),color=['green','blue','red','purple','orange'])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

best_model_name= max(accuracy_results,key=accuracy_results.get)
best_model = models[best_model_name]
best_accuracy = accuracy_results[best_model_name]
print(f"Best Model: {best_model}")
print(f"Best Accuracy: {best_accuracy}")

