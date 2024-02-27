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
# # model.add(Dense(16, activation='relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dropout(0.3))


# model.add(Dense(5, activation='softmax')) #output is the number of different categories we are trying to calculate between 

# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# labels = ['coffee', 'irishcream', 'kahlua', 'rum', 'test']

# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
# model.fit(X_train_reshape, y_train_one_hot, epochs=250, batch_size=8, validation_data=(X_valid_reshape, y_valid_one_hot), callbacks=[early_stop, reduce_lr])

# model.save("PoCmodel.h5")

## TESTING ###

def normalizeData(mins, ranges, raw_data):
  return (raw_data - mins) / ranges

mins = np.array([16456.6, 16457.8, 12055.9, 12051.2, 12050.1, 10586.2, 10582.7, 10581.9, 15426.9, 15426.8, 15423.5, 7419.4, 7418.4, 7418.2, 20077.8, 20073.2, 27028.0, 27000.3, 7689.3, 7688.6, 7688.2, 6639.7, 6640.9, 6641.3, 12181.1, 12181.3, 12178.3, 22771.9, 22777.8, 22776.5, 18968.4, 18964.8, 35323.0, 35311.1, 26581.3, 26569.1, 26546.4, 94231.5, 94230.0, 94321.9, 52542.7, 52518.6, 52475.5, 64815.3, 64845.7, 64867.6, 116893.2, 116778.5, 1480640.8, 1524820.8, 36167.1, 36186.9, 36183.8, 14611.9, 14611.9, 14611.4, 1092.5, 1092.5, 3815.4, 3817.8, 3819.1, 25619.3, 25616.0, 22.63, 36.06])

ranges = np.array([5233.3, 5235.3, 31352.0, 31405.4, 31407.8, 95530.4, 95782.4, 95643.6, 7959.1, 7955.3, 7959.5, 17726.2, 17720.3, 17718.6, 2963.4, 2968.0, 7679.8, 7627.0, 6942.7, 6944.8, 6943.9, 40865.6, 40890.4, 40898.4, 77095.4, 77041.6, 77066.7, 100566.7, 100411.4, 100462.5, 15347.9, 15376.1, 9433.4, 9408.1, 44024.5, 44495.8, 44169.4, 253968.8, 254031.4, 253207.5, 84372.3, 84405.9, 84442.7, 264897.2, 264456.4, 265311.3, 17935.2, 18212.0, 433117.4, 283712.8, 80863.0, 80881.2, 80827.8, 13840.5, 13841.5, 13834.7, 0.1, 0.1, 19949.9, 19984.4, 20006.4, 13795.5, 13784.7, 6.04, 30.2])

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
for i in range(10):
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
  class_labels = ['coffee', 'irishcream', 'kahlua', 'rum', 'test']
  predicted_class_labels = [class_labels[label_index] for label_index in predicted_labels]

  print("Predicted class labels:", predicted_class_labels)


  

