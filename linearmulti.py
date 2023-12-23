import pandas as pd
from word2number import w2n
from sklearn.linear_model import LinearRegression
import math

df = pd.read_csv('datasets\hiring.csv')


df.experience = df.experience.fillna("zero")
df.experience = df.experience.apply(w2n.word_to_num)
median_test_score = math.floor(df['test_score(out of 10)'].mean())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)


model = LinearRegression()
model.fit(df[['experience','test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])


twoyr = model.predict([[2,9,6]])
twelveyr = model.predict([[12,10,10]])

print(twoyr)
print(twelveyr)