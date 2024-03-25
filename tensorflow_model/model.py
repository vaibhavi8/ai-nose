import csv
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from keras.metrics import Accuracy, Precision, Recall
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
#### Neural Network ####
# ### FLATTENING DATA ###
tf.compat.v1.disable_eager_execution()
LABELS = ["coffee", "sandalwood", "unknown", "fent"]

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

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)
# y_train_encoded= np.eye(len(set(y_train)))[y_train]
# y_valid_encoded= np.eye(len(set(y_valid)))[y_valid]

encoder = OneHotEncoder()

y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_valid_one_hot = encoder.transform(y_valid.reshape(-1, 1)).toarray()

print(X_train.shape)
print(X_valid.shape)

### DEFINING MODEL STRUCTURE ###
model = Sequential()
# model.add(Flatten(input_shape=(66,5))) #flattening data by defining there are 66 sensors and 5 features

# model.add(Dense(128, activation='relu')) #hidden layer 1
# model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))  #hidden layer 2
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.2))


model.add(Dense(len(LABELS), activation='softmax', kernel_regularizer='l2')) #output is the number of different categories we are trying to calculate between 

model.compile(optimizer=Adam(learning_rate=0.005), loss='categorical_crossentropy', metrics=["accuracy"])


early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.20, patience=2, min_lr=0.0001)
model.fit(X_train, y_train_one_hot, epochs=130, batch_size=8, validation_data=(X_valid, y_valid_one_hot), callbacks=[reduce_lr, early_stop] ) #callbacks=[early_stop, reduce_lr]

model.save("PoCmodel.h5")

### DECISION TREE###




clf = DecisionTreeClassifier(max_leaf_nodes=12, random_state=42)

clf.fit(X_train, y_train)

y_pred_encoded = clf.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred_encoded)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_valid, y_pred_encoded))

X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
X_valid_df = pd.DataFrame(X_valid, columns=[f'feature_{i}' for i in range(X_valid.shape[1])])

class_names_list = encoder.categories_[0].tolist()

joblib.dump(clf, "TreeNoTorH.pkl")
# print(class_names_list)

# Visualize the decision tree (optional)
plt.figure(figsize=(14, 8))
plot_tree(clf, filled=True, feature_names=X_train_df.columns.tolist(), class_names=class_names_list)
plt.show()



###  RANDOM FOREST ###

clf = RandomForestClassifier(n_estimators=3, max_depth=4, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_valid, y_pred))

joblib.dump(clf, 'randomForestModel.pkl')


# X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
# X_valid_df = pd.DataFrame(X_valid, columns=[f'feature_{i}' for i in range(X_valid.shape[1])])

# # Choose a specific tree from the forest (e.g., the first tree)
# tree_to_visualize = clf.estimators_[1]

# # Visualize the chosen tree
# plt.figure(figsize=(14, 8))
# tree.plot_tree(tree_to_visualize, filled=True, feature_names=X_train_df.columns.tolist(), class_names=LABELS)
# plt.show()


  ### K-NEAREST NEIGHBORS ###

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)


# Feature selection
selector = SelectKBest(score_func=f_classif, k=7)
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

# # Assuming X_train and y_train_one_hot are your training data and one-hot encoded labels
# scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(y_train_one_hot, axis=1), cmap=plt.cm.coolwarm)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Scatter Plot of Features 1 and 2')

# # Create a colorbar
# cbar = plt.colorbar(scatter)
# cbar.set_label('Labels')

# # Annotate the colorbar with class labels
# cbar.set_ticks(np.arange(len(LABELS)))
# cbar.set_ticklabels(LABELS)

# # Show the plot
# plt.show()




# Plotting every combination of features
num_features = len(selected_features)
num_plots = num_features * (num_features - 1) // 2  # Calculate total number of plots

plt.figure(figsize=(15, 10))
plot_index = 1

for i in range(num_features):
    for j in range(i + 1, num_features):
        plt.subplot(num_features - 1, num_features - 1, plot_index)  # Create subplot
        for label in np.unique(y_valid):
            plt.scatter(X_valid_scaled[y_valid == label, i], X_valid_selected[y_valid == label, j], label=label)
        plt.xlabel('Feature {}'.format(selected_features[i]))
        plt.ylabel('Feature {}'.format(selected_features[j]))
        plt.title('Relationship between Feature {} and Feature {}'.format(selected_features[i], selected_features[j]))
        plt.legend()
        plot_index += 1

plt.tight_layout()
plt.show()

