from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels = ["red", "green", "blue", "green", "white"]

# Fit the encoder and transform the data
le.fit(labels)
transformed_labels = le.transform(labels)

# Result: transformed_labels = [2, 1, 0, 1]

print(list(le.classes_))
print(transformed_labels)