import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


df = pd.read_csv("datasets/titanic.csv")
mean_age = df['Age'].mean()
df['Age'].fillna(mean_age, inplace=True)

drop = ['Name', 'Survived', 'SibSp', 'Parch','Ticket','Cabin', 'Embarked', 'PassengerId']
inputs = df.drop(columns=drop, axis='columns')

target = df['Survived']

le_class = LabelEncoder()
le_Sex = LabelEncoder()
le_Age = LabelEncoder()
le_Fare = LabelEncoder()

inputs['Pclass_n'] = le_class.fit_transform(inputs['Pclass'])
inputs['Sex_n'] = le_Sex.fit_transform(inputs['Sex'])
inputs['Age_n'] = le_Age.fit_transform(inputs['Age'])
inputs['Fare_n'] = le_Fare.fit_transform(inputs['Fare'])


drop_n = ['Pclass', 'Sex', 'Age', 'Fare']
inputs_n = inputs.drop(columns=drop_n, axis='columns')


model = tree.DecisionTreeClassifier()

model.fit(inputs_n, target)




print(inputs.head(10))
print(model.score(inputs_n,target))
print(dir(df))