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

## TESTING ###

def normalizeData(mins, ranges, raw_data):
  return (raw_data - mins) / ranges
mins = np.array([13021.2, 13027.7, 9867.3, 9870.7, 9867.1, 8382.5, 8379.6, 8377.7, 11807.0, 11804.2, 11804.2, 8243.1, 8241.2, 8243.3, 18508.6, 18505.4, 21621.9, 21609.9, 5272.8, 5272.4, 5272.5, 2727.5, 2727.5, 2728.7, 9318.4, 9315.8, 9313.4, 25743.0, 25742.2, 25748.3, 14938.4, 14935.5, 32740.5, 32735.1, 30230.2, 30272.1, 30265.7, 51209.7, 51264.1, 51279.1, 47878.9, 47856.3, 47904.2, 29830.5, 29883.2, 29892.7, 86227.5, 86020.4, 1533263.9, 1565930.4, 25931.8, 25887.3, 25926.9, 12724.3, 12724.8, 12722.8, 1092.5, 1092.5, 2733.8, 2735.4, 2737.4, 24254.1, 24256.2, 27.84, 34.92])

ranges = np.array([1463.9, 1457.7, 8902.5, 8905.8, 8906.2, 8706.0, 8709.1, 8711.1, 3893.9, 3895.1, 3896.4, 13364.4, 13371.1, 13367.1, 1464.1, 1463.6, 3315.4, 3324.1, 2054.1, 2054.5, 2054.9, 4897.1, 4898.7, 4899.7, 16682.2, 16680.9, 16676.9, 41299.1, 41284.4, 41228.1, 2833.3, 2829.3, 6146.6, 6189.6, 19161.1, 19144.5, 19121.9, 48275.1, 48230.7, 48277.3, 75930.7, 75890.3, 75754.9, 52462.9, 52508.2, 52498.7, 12612.6, 12334.3, 498372.6, 457764.5, 21242.9, 21290.2, 21287.7, 4919.8, 4916.6, 4937.9, 0.1, 0.1, 2198.6, 2199.7, 2200.2, 5678.0, 5681.5, 3.12, 25.88])

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
for i in range(16):
  testFilePath = os.path.join('../testing/',os.listdir('../testing/')[i])
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
  class_labels = ["coffee", "orange", "sandalwood", "unknown"]
  predicted_class_labels = [class_labels[label_index] for label_index in predicted_labels]

  print("Predicted class labels:", predicted_class_labels)