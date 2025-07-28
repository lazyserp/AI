import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("student_scores.csv")

X =df[['Hours_Studied']] #input for training model
y = df['Marks'] # predicitng marks based on hours studied

#split data 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.legend()
plt.title("Linear Regression - Marks Prediction")
plt.show()
