import numpy as np
from sklearn.svm import SVC

# Given input patterns X and output patterns y
X = np.array([[1,2,3],[3,4,5],[5,6,7],[8,9,10]])
y = np.array([13,12,11,10])  # Assuming binary class labels

# Create an SVM classifier with an "rbf" kernel and probability estimates enabled
clf = SVC(kernel='rbf', probability=True)

# Fit the SVM model to your data
clf.fit(X, y)

# Determine probability estimates
probability_estimates = clf.predict_proba(X)

print("Probability estimates:\n", probability_estimates)
