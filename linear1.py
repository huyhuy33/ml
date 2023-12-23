import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets\canada_per_capita_income.csv')

X = df[['year']]
y = df['per capita income (US$)']


model = LinearRegression()

model.fit(X, y)

inter = model.intercept_
coef = model.coef_[0]

predict = coef*2020+inter

print(inter)
print(coef)

print(predict)