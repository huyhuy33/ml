import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


train = pd.read_csv('spam.csv', encoding='latin1')
test = pd.read_csv('test.csv', encoding='latin1')

train.loc[train['Category'] == 'spam', 'Category'] = 0
train.loc[train['Category'] == 'ham', 'Category'] = 1
test.loc[test['Category'] == 'spam', 'Category'] = 0
test.loc[test['Category'] == 'ham', 'Category'] = 1
#
# X = train.values[:,0]
# Y = train.values[:,1]
# x = test.values[:,0]
# y = test.values[:,1]

X = train['Category']
Y = train['Message']
x = test['Category']
y = test['Message']

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

train_features = feature_extraction.fit_transform(Y)
test_features = feature_extraction.transform(y)

X = X.astype('int')
x = x.astype('int')

model = LogisticRegression()
model.fit(train_features,X)

# input_ur_mail = [y]
# input_data = feature_extraction.transform(input_ur_mail)
# pred = model.predict(input_data)


predictions = model.predict(test_features)

# Now 'predictions' contains the predicted labels for X_test
print(predictions)


