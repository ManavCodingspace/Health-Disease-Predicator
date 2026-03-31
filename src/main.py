import pandas as pd
import numpy as np 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings


warnings.filterwarnings('ignore')

health = pd.read_csv('heart.csv')
health.columns = health.columns.str.strip()
print(health.columns)

print(health.info())
print(health.head())

columns = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
x = health.drop(['HeartDisease'], axis = 1)
y = health['HeartDisease']
print(x.shape, y.shape)

x_encoded = pd.get_dummies(x, columns=columns, drop_first = True).astype(int)
print(x_encoded.head())

x_train, x_test, y_train, y_test = train_test_split(x_encoded, y , test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

models ={
    "logistic Regression" : LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Naive Baies': GaussianNB(),
    'Decision Tree' : DecisionTreeClassifier(),
    'SVM': SVC()
}
result = []

for name,model in models.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    result.append({
        'model':name,
        'Accuracy':round(acc,4),
        'F1_score':round(f1,4)
    })
print(result)

joblib.dump(models['SVM'], 'svm_heart.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(x_encoded.columns.tolist(), 'column.pkl')