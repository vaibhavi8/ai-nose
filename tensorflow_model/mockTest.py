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
import joblib
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
## TESTING ###

def normalizeData(mins, ranges, raw_data):
  return (raw_data - mins) / ranges
mins = np.array([8537.3, 8537.2, 3977.6, 3976.2, 3977.1, 2210.4, 2210.3, 2211.2, 6330.0, 6329.7, 6330.2, 2877.2, 2876.9, 2876.9, 13131.9, 13132.3, 13449.4, 13426.5, 2514.1, 2514.0, 2514.0, 1502.6, 1502.2, 1502.9, 4454.6, 4454.6, 4454.8, 9418.5, 9417.1, 9417.7, 8384.5, 8384.3, 25443.8, 25445.9, 15045.3, 15048.7, 15041.6, 23136.3, 23147.7, 23174.4, 17980.3, 17977.5, 17970.7, 9858.7, 9860.5, 9864.4, 64656.6, 64673.9, 1477884.1, 1485443.4, 18127.2, 18128.5, 18129.8, 6693.5, 6692.8, 6692.7, 1092.5, 1092.5, 1743.4, 1743.9, 1744.6, 14600.0, 14605.5, 22.38, 30.99])

ranges = np.array([8126.1, 8123.6, 22705.4, 22688.2, 22699.5, 40997.3, 40972.3, 41012.4, 21101.0, 21095.1, 21092.0, 18730.3, 18735.4, 18733.5, 8591.0, 8584.0, 14465.4, 14469.0, 7455.1, 7454.2, 7454.5, 12641.5, 12648.1, 12652.1, 39860.1, 39860.1, 39846.7, 58464.9, 58495.7, 58459.9, 14405.7, 14399.4, 18904.2, 18892.6, 1830203.8, 1804261.8, 34346.0, 123724.9, 123695.4, 123379.3, 132172.0, 131629.2, 131728.3, 72434.7, 72530.9, 72527.0, 54750.7, 55119.3, 553752.4, 511775.1, 62480.1, 62525.8, 62516.9, 17831.9, 17814.2, 17803.8, 0.1, 0.1, 7329.9, 7338.0, 7342.3, 20829.4, 20882.1, 8.58, 29.81])

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
  
class_labels = ["coffee", "sandalwood", "unknown"]# loaded_model = [load_model("models/PoCmodel.h5"), "NN"] #neural network
# loaded_model = [joblib.load('models/logisticRegression.pkl'), "LogReg"] #logistic regression
# loaded_model = [joblib.load('models/GradientBoosted.pkl'), "GB"] #gradient boosted
loaded_model = [joblib.load('models/DecisionTreeModel.pkl'), "Tree"] #decision tree
# loaded_model = [joblib.load('models/randomForestModel.pkl'), "RF"] #random forest
# loaded_model =[ joblib.load('models/knnModel.pkl'), "KNN"] #K-Nearest Neighbors
# loaded_model = [joblib.load('models/TreeNoTorH.pkl'), "TreeNoT"] #decision tree without temperature or humidity
model = loaded_model[0]
for i in range(15):
  testFilePath = os.path.join('../testing/',os.listdir('../testing/')[i])
  print(testFilePath)
  data = []
  with open(testFilePath, 'r') as f:
    csv_reader = csv.reader(f, delimiter=';')
    data = next(csv_reader)[1:]
  data = np.array(data).astype(float)
  prepped_data = preprocessing(data)
  prepped_data = prepped_data.reshape(1, -1)
  predictions = loaded_model[0].predict(prepped_data)

  if loaded_model[1] == "NN" or loaded_model[1] == "KNN":
      print(predictions)
      predicted_labels = np.argmax(predictions, axis=1)
      predicted_class_labels = [class_labels[label_index] for label_index in predicted_labels]
      print("Predicted class labels:", predicted_class_labels)
  else:
      print("Predicted Label:", class_labels[predictions[0]])

  # 'predictions' will contain the predicted probabilities for each class
  # If you want to get the predicted class label, you can use argmax to find the index of the class with the highest probability
  # predicted_labels = np.argmax(predictions, axis=1)

  # If you have class labels, you can map the predicted label indices back to their corresponding class labels

  # predicted_class_labels = [class_labels[label_index] for label_index in predicted_labels]

  # print("Predicted class labels:", predicted_class_labels)