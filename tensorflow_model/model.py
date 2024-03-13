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

# # # # ### FLATTENING DATA ###

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
# # model.add(Dense(16, activation='relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dropout(0.3))


# model.add(Dense(3, activation='softmax')) #output is the number of different categories we are trying to calculate between 

# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# # labels = ['coffee', 'orange', 'kahlua', 'rum', 'test']
# labels = ['coffee', 'orange', 'unknown']

# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=0.0001)
# model.fit(X_train_reshape, y_train_one_hot, epochs=50, batch_size=5, validation_data=(X_valid_reshape, y_valid_one_hot), callbacks=[early_stop, reduce_lr])

# model.save("PoCmodel.h5")

# TESTING ###

def normalizeData(mins, ranges, raw_data):
  return (raw_data - mins) / ranges

mins = np.array([11832.6, 11830.7, 6478.0, 6479.4, 6479.2, 4892.6, 4891.9, 4892.3, 8149.5, 8148.9, 8148.5, 4512.8, 4512.4, 4512.8, 17366.4, 17366.9, 20199.8, 20162.3, 4134.1, 4133.9, 4133.8, 2206.6, 2206.5, 2207.5, 7679.4, 7675.3, 7676.4, 16481.3, 16485.3, 16491.2, 13152.2, 13148.5, 29304.1, 29300.7, 21153.4, 21152.0, 21130.4, 33166.8, 33225.9, 33236.4, 29919.6, 29869.3, 29866.7, 20853.9, 20852.0, 20867.9, 84672.6, 84737.7, 1631367.4, 1548417.5, 23295.6, 23293.6, 23296.7, 10194.8, 10192.8, 10194.6, 1092.5, 1092.5, 2058.3, 2058.5, 2059.7, 18641.1, 18652.2, 27.77, 30.86])

ranges = np.array([1192.7, 1197.8, 3652.5, 3652.9, 3644.7, 1500.2, 1498.1, 1495.5, 1288.8, 1289.5, 1288.7, 5180.9, 5180.9, 5180.3, 888.6, 896.1, 2257.4, 2256.9, 472.6, 473.1, 474.3, 1015.4, 1015.8, 1015.8, 4150.4, 4160.6, 4152.6, 11670.7, 11634.4, 11651.3, 1546.8, 1548.6, 2090.4, 2117.6, 5048.8, 5082.1, 5104.5, 9802.9, 9841.6, 9810.3, 9384.9, 9406.0, 9381.7, 6160.6, 6154.5, 6134.8, 15379.5, 15533.7, 358508.1, 346760.0, 2363.3, 2380.6, 2388.1, 176.1, 175.0, 172.0, 0.1, 0.1, 214.9, 215.4, 215.2, 1890.2, 1886.8, 0.94, 28.84])

# print(len(mins))
# print(len(ranges))

def preprocessing(raw_data): #data type = np array
  header = len(raw_data)

   #we want it to only check for dropped columns
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
  # Initialize preprocessed data array
  prep_data = np.array([])  # Initialize as an empty 1D array

  # Iterate over columns and preprocess data
  for i in range(len(raw_data)):
      # Drop column if requested
      if preproc[i] == PREP_DROP:
          # print("Dropping column", i+1)
          continue

      # Otherwise, append the column value to the preprocessed data
      prep_data = np.append(prep_data, raw_data[i])
   
  return normalizeData(mins, ranges, prep_data)
  
loaded_model = load_model("PoCmodel.h5")
for i in range(18):
  testFilePath = os.path.join('output/testing/',os.listdir('output/testing/')[i])
  print(testFilePath)
  data = []
  with open(testFilePath, 'r') as f:
    csv_reader = csv.reader(f, delimiter=';')
    data = next(csv_reader)[1:]
  data = np.array(data).astype(float)
  prepped_data = preprocessing(data)
  prepped_data = prepped_data.reshape(1, -1)

  predictions = loaded_model.predict(prepped_data)
  # print(predictions.max()) use this to get the probability of this label.

  print(predictions)

  # 'predictions' will contain the predicted probabilities for each class
  # If you want to get the predicted class label, you can use argmax to find the index of the class with the highest probability
  predicted_labels = np.argmax(predictions, axis=1)

  # If you have class labels, you can map the predicted label indices back to their corresponding class labels
  class_labels = ['coffee', 'orange', 'unknown']
  predicted_class_labels = [class_labels[label_index] for label_index in predicted_labels]

  print("Predicted class labels:", predicted_class_labels)


  

