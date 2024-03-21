import csv
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import joblib

LABELS = ["coffee", "sandalwood","unknown"]

# gathering preprocessed data from training/validation/testing file
trainingPath = 'output/train/'
validPath = 'output/val/'

def flatten(path):
    features_dirs = []
    data = []
    labels = []
    for dir in os.listdir(path):
        features_dirs.append(dir)
        for file in os.listdir(os.path.join(path, dir)):
            filepath = os.path.join(path, dir, file)
            with open(filepath, newline='') as afile:
                csvreader = csv.reader(afile)
                next(csvreader)
                for row in csvreader:
                    labels.append(dir)
                    data.append(row)
    data = np.array(data).astype(float)
    labels = np.array(labels)

    return data, labels

X_train, y_train = flatten(trainingPath)
X_valid, y_valid = flatten(validPath)

encoder = OneHotEncoder()

y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_valid_one_hot = encoder.transform(y_valid.reshape(-1, 1)).toarray()

clf = DecisionTreeClassifier(max_leaf_nodes=4, random_state=42)

clf.fit(X_train, y_train)

y_pred_encoded = clf.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred_encoded)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_valid, y_pred_encoded))

X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
X_valid_df = pd.DataFrame(X_valid, columns=[f'feature_{i}' for i in range(X_valid.shape[1])])

class_names_list = encoder.categories_[0].tolist()

joblib.dump(clf, "DecisionTreeModel.pkl")
# print(class_names_list)

# Visualize the decision tree (optional)
plt.figure(figsize=(14, 8))
plot_tree(clf, filled=True, feature_names=X_train_df.columns.tolist(), class_names=class_names_list)
plt.show()
