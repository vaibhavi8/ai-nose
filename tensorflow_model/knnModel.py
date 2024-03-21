
import csv
import os
import keras
import tensorflow as tf 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
tf.compat.v1.disable_eager_execution()
LABELS = ["coffee", "sandalwood", "unknown"]

#gathering preprocessed data from training/validation/testing file
trainingPath = 'output/train/'
validPath = 'output/val/'
testPath = 'output/test'
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
        features = next(csvreader)
        for row in csvreader:
          labels.append(dir)
          data.append(row)
  data = np.array(data).astype(float)
  labels = np.array(labels)

  return data, labels, features

X_train, y_train, features = flatten(trainingPath)
X_valid, y_valid, ig = flatten(validPath)

print("Num Classes:", len(LABELS))

encoder = OneHotEncoder()

y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_valid_one_hot = encoder.transform(y_valid.reshape(-1, 1)).toarray()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)


# Feature selection
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X_train_scaled, y_train)
selected_features = np.array(features)[selector.get_support(indices=True)]  # Get names of selected features
print("Selected Features:", selected_features)

X_train_selected = selector.transform(X_train_scaled)
X_valid_selected = selector.transform(X_valid_scaled)

knn = KNeighborsClassifier(n_neighbors=80)

knn.fit(X_train_selected, y_train_one_hot)

y_pred = knn.predict(X_valid_selected)

accuracy = accuracy_score(y_valid_one_hot, y_pred)
print("Accuracy: ", accuracy)

joblib.dump(knn, 'knnModel.pkl')


# # Plotting every combination of features
# num_features = len(selected_features)
# num_plots = num_features * (num_features - 1) // 2  # Calculate total number of plots

# plt.figure(figsize=(15, 10))
# plot_index = 1

# for i in range(num_features):
#     for j in range(i + 1, num_features):
#         plt.subplot(num_features - 1, num_features - 1, plot_index)  # Create subplot
#         for label in np.unique(y_valid):
#             plt.scatter(X_valid_selected[y_valid == label, i], X_valid_selected[y_valid == label, j], label=label)
#         plt.xlabel('Feature {}'.format(selected_features[i]))
#         plt.ylabel('Feature {}'.format(selected_features[j]))
#         plt.title('Relationship between Feature {} and Feature {}'.format(selected_features[i], selected_features[j]))
#         plt.legend()
#         plot_index += 1

# # plt.tight_layout()
# plt.show()


