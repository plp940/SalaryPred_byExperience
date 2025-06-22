#project of a simple salary prediction model application based on years of experience
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
#import pickle, joblib
import joblib
import pickle

df = pd.read_csv('salary_data.csv')
#print(df.info())

#split the dat into independent variables and target variables
X = df[["YearsExperience"]] # Independent variable (Years of Experience)
y = df[["Salary"]]  # Dependent variable (Salary)
#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#scale down the data feature is important for some models,as every feature should be on the same scale
#normalisation or scaling is not required for linear regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Check the shape of the datasets
#print("Training set shape:", X_train.shape, y_train.shape)
#print("Testing set shape:", X_test.shape, y_test.shape)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model using joblib
joblib.dump(model, 'predict_salary.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")
