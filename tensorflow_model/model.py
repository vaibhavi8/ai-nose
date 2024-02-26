import csv
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from sklearn import metrics

# ### FLATTENING DATA ###

# #gathering preprocessed data from training/validation/testing file
# trainingPath = 'output/train/'
# validPath = 'output/val/'
# testPath = 'output/test'
# def flatten(path):
#   features_dirs = []
#   data = []
#   labels = []
#   for dir in os.listdir(path):
#     features_dirs.append(dir)
#     for file in os.listdir(os.path.join(path, dir)):
#       filepath = os.path.join(path, dir, file)
#       with open(filepath, newline='') as afile:
#         csvreader = csv.reader(afile)
#         next(csvreader)
#         for row in csvreader:
#           labels.append(file[0:3])
#           data.append(row)
#   data = np.array(data).astype(float)
#   labels = np.array(labels)

#   return data, labels

# X_train, y_train = flatten(trainingPath)
# X_valid, y_valid = flatten(validPath)

# label_encoder = LabelEncoder()

# y_train_encoded = label_encoder.fit_transform(y_train)
# y_valid_encoded = label_encoder.transform(y_valid)


# num_classes = len(label_encoder.classes_)

# y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
# y_valid_one_hot = to_categorical(y_valid_encoded, num_classes=num_classes)


# X_train_reshape = X_train.reshape(-1, 65) #make changes based on features and sensors
# X_valid_reshape = X_valid.reshape(-1, 65) #make changes based on features and sensors


# ### DEFINING MODEL STRUCTURE ###
# model = Sequential()
# # model.add(Flatten(input_shape=(66,5))) #flattening data by defining there are 66 sensors and 5 features

# model.add(Dense(64, activation='relu', input_shape=(65,))) #hidden layer 1
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))  #hidden layer 2
# model.add(Dropout(0.3))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.3))


# model.add(Dense(2, activation='softmax')) #output is the number of different categories we are trying to calculate between 

# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# labels = ['coffee', 'rum']

# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001)
# model.fit(X_train_reshape, y_train_one_hot, epochs=50, batch_size=8, validation_data=(X_valid_reshape, y_valid_one_hot), callbacks=[early_stop, reduce_lr])

# model.save("PoCmodel.h5")

## TESTING ###

def preprocessing(raw_data): #data type = np array
  # Calculate means, standard deviations, and ranges
  means = np.mean(raw_data, axis=0)
  std_devs = np.std(raw_data, axis=0)
  maxes = np.max(raw_data, axis=0)
  mins = np.min(raw_data, axis=0)
  ranges = np.ptp(raw_data, axis=0)
  header = len(raw_data)
  PREP_DROP = -1
  PREP_NONE = 0 
  PREP_STD = 1
  PREP_NORM = 2

  preproc = [PREP_NORM,   # 1
           PREP_NORM,   # 2
           PREP_NORM,   # 3
           PREP_NORM,   # 4
           PREP_NORM,   # 5
           PREP_NORM,   # 6
           PREP_NORM,   # 7
           PREP_NORM,   # 8
           PREP_NORM,   # 9
           PREP_NORM,   # 10
           PREP_NORM,   # 11
           PREP_NORM,   # 12
           PREP_NORM,   # 13
           PREP_NORM,   # 14
           PREP_NORM,   # 15
           PREP_NORM,   # 16
           PREP_NORM,   # 17
           PREP_NORM,   # 18
           PREP_NORM,   # 19
           PREP_NORM,   # 20
           PREP_NORM,   # 21
           PREP_NORM,   # 22
           PREP_NORM,   # 23
           PREP_NORM,   # 24
           PREP_NORM,   # 25
           PREP_NORM,   # 26
           PREP_NORM,   # 27
           PREP_NORM,   # 28
           PREP_NORM,   # 29
           PREP_NORM,   # 30
           PREP_NORM,   # 31
           PREP_NORM,   # 32
           PREP_NORM,   # 33
           PREP_NORM,   # 34
           PREP_NORM,   # 35
           PREP_NORM,   # 36
           PREP_NORM,   # 37
           PREP_NORM,   # 38
           PREP_NORM,   # 39
           PREP_NORM,   # 40
           PREP_NORM,   # 41
           PREP_NORM,   # 42
           PREP_NORM,   # 43
           PREP_NORM,   # 44
           PREP_NORM,   # 45
           PREP_NORM,   # 46
           PREP_NORM,   # 47
           PREP_NORM,   # 48
           PREP_NORM,   # 49
           PREP_NORM,   # 50
           PREP_NORM,   # 51
           PREP_NORM,   # 52
           PREP_NORM,   # 53
           PREP_NORM,   # 54
           PREP_NORM,   # 55
           PREP_NORM,   # 56
           PREP_DROP,   # 57
           PREP_NORM,   # 58
           PREP_NORM,   # 59
           PREP_NORM,   # 60
           PREP_NORM,   # 61
           PREP_NORM,   # 62
           PREP_NORM,   # 63
           PREP_NORM,   # 64
           PREP_NORM,   # temperature
           PREP_NORM]   #humidity
  assert(len(preproc)==raw_data.shape[0])


  num_cols = sum(1 for x in preproc if x != PREP_DROP)
  prep_data = np.zeros((1, num_cols))
  prep_means = []
  prep_std_devs = []
  prep_mins = []
  prep_ranges = []

    # Go through each column to preprocess the data
  prep_c = 0
  for i in range(len(raw_data)):

    # Drop column if requested
    if preproc[i] == PREP_DROP:
        print("Dropping", i+1)
        continue

    # Perform data standardization
    if preproc[i] == PREP_STD:
        prep_data[0, prep_c] = (raw_data[i] - means) / std_devs

    # Perform data normalization
    elif preproc[i] == PREP_NORM:
        prep_data[0, prep_c] = (raw_data[i] - mins) / ranges

    # Copy data over if no preprocessing is requested
    elif preproc[i] == PREP_NONE:
        prep_data[0, prep_c] = raw_data[i]

    # Error if code not recognized
    else:
        raise Exception("Preprocessing code not recognized")

  # Copy header (and preprocessing constants) and increment preprocessing column index
    prep_means.append(means)
    prep_std_devs.append(std_devs)
    prep_mins.append(mins)
    prep_ranges.append(ranges)
    prep_c += 1

  return prep_data
loaded_model = load_model("PoCmodel.h5")
for i in range(8):
  testFilePath = os.path.join('output/testing/',os.listdir('output/testing/')[i])
  print(testFilePath)
  data = []
  with open(testFilePath, 'r') as f:
    csv_reader = csv.reader(f, delimiter=';')
    data = next(csv_reader)[1:]
  data = np.array(data).astype(float)
  prepped_data = preprocessing(data)
  predictions = loaded_model.predict(prepped_data)
  print(predictions)

  # 'predictions' will contain the predicted probabilities for each class
  # If you want to get the predicted class label, you can use argmax to find the index of the class with the highest probability
  predicted_labels = np.argmax(predictions, axis=1)

  # If you have class labels, you can map the predicted label indices back to their corresponding class labels
  class_labels = ['coffee', 'rum']
  predicted_class_labels = [class_labels[label_index] for label_index in predicted_labels]

  print("Predicted class labels:", predicted_class_labels)


  

