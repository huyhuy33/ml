from sklearn.datasets import load_digits
import pandas as pd
digits = load_digits()

df = pd.DataFrame(digits.data, digits.target)
df['target'] = digits.target



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3)

from sklearn.svm import SVC
# rbf_model = SVC(kernel='rbf')
# rbf_model.fit(X_train, y_train)

linear_model = SVC(kernel='linear')
linear_model.fit(X_train,y_train)

print(linear_model.score(X_test,y_test))