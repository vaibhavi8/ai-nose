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

def normalizeData(mins, ranges, raw_data):
  return (raw_data - mins) / ranges

mins = np.array([17574.5, 17566.8, 30403.4, 30394.0, 30411.1, 23203.0, 23228.2, 23252.9, 19406.1, 19402.1, 19395.6, 12410.9, 12411.1, 12407.8, 20933.8, 20928.1, 27028.0, 27000.3, 8457.5, 8458.0, 8459.2, 13884.2, 13889.3, 13886.2, 23073.8, 23059.4, 23047.7, 39916.7, 39889.8, 39894.5, 18968.4, 18964.8, 40654.2, 40650.3, 34637.4, 34656.7, 34618.0, 191146.8, 191325.0, 190965.8, 93533.5, 93496.0, 93418.9, 91400.4, 91433.4, 91361.8, 118789.8, 118813.5, 1550433.8, 1524820.8, 48062.5, 48098.9, 48109.6, 16729.0, 16729.2, 16731.6, 1092.5, 1092.5, 5061.7, 5066.3, 5070.6, 27324.3, 27363.9, 24.71, 41.39])

ranges = np.array([710.3, 704.5, 13004.5, 13062.6, 13046.8, 13818.7, 13773.2, 13747.7, 1110.4, 1117.8, 1111.5, 5403.8, 5407.1, 5402.4, 175.5, 187.3, 285.6, 277.8, 180.2, 177.5, 177.2, 1807.8, 1803.8, 1814.2, 6019.7, 6030.6, 6031.1, 12391.8, 12411.1, 12401.6, 444.2, 441.6, 903.5, 898.6, 6506.0, 6537.2, 6592.3, 19690.2, 19258.2, 20031.9, 11784.6, 12026.6, 12814.2, 20947.3, 20931.3, 20875.6, 2448.7, 2572.2, 217288.7, 178177.7, 9254.2, 9116.0, 9153.3, 603.4, 611.1, 603.2, 0.1, 0.1, 551.5, 551.2, 549.7, 1776.3, 1754.4, 2.38, 24.87])

print(len(mins))
print(len(ranges))

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
          print("Dropping column", i+1)
          continue

      # Otherwise, append the column value to the preprocessed data
      prep_data = np.append(prep_data, raw_data[i])
   
  return normalizeData(mins, ranges, prep_data)
  
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
  prepped_data = prepped_data.reshape(1, -1)

  predictions = loaded_model.predict(prepped_data)
  print(predictions)

  # 'predictions' will contain the predicted probabilities for each class
  # If you want to get the predicted class label, you can use argmax to find the index of the class with the highest probability
  predicted_labels = np.argmax(predictions, axis=1)

  # If you have class labels, you can map the predicted label indices back to their corresponding class labels
  class_labels = ['coffee', 'rum']
  predicted_class_labels = [class_labels[label_index] for label_index in predicted_labels]

  print("Predicted class labels:", predicted_class_labels)


  

