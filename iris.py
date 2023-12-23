from sklearn import datasets
import pandas as pd


iris = datasets.load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
target = iris.target
data = iris.data
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])


setosa = data[target == 0]
versicolor = data[target == 1]
virginica = data[target == 2]




print(setosa.shape)