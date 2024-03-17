import csv
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import Accuracy, Precision, Recall
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from sklearn import metrics

# ### FLATTENING DATA ###

LABELS = ["coffee", "sandalwood", "orange", "unknown"]

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
        next(csvreader)
        for row in csvreader:
          labels.append(file[0:3])
          data.append(row)
  data = np.array(data).astype(float)
  labels = np.array(labels)

  return data, labels

X_train, y_train = flatten(trainingPath)
X_valid, y_valid = flatten(validPath)

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)

num_classes = len(label_encoder.classes_)

y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
y_valid_one_hot = to_categorical(y_valid_encoded, num_classes=num_classes)

print(X_train.shape)
print(X_valid.shape)

### DEFINING MODEL STRUCTURE ###
model = Sequential()
# model.add(Flatten(input_shape=(66,5))) #flattening data by defining there are 66 sensors and 5 features

# model.add(Dense(128, activation='relu')) #hidden layer 1
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))  #hidden layer 2
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.2))


model.add(Dense(len(LABELS), activation='softmax')) #output is the number of different categories we are trying to calculate between 

model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=["accuracy"])


# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.20, patience=2, min_lr=0.0001)
model.fit(X_train, y_train_one_hot, epochs=130, batch_size=8, validation_data=(X_valid, y_valid_one_hot), callbacks=[reduce_lr] ) #callbacks=[early_stop, reduce_lr]

model.save("PoCmodel.h5")




  

