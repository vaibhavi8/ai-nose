#/usr/bin/env python3
import csv
import argparse
import os
import time
import numpy as np
import keras
from keras.models import load_model

from sklearn.preprocessing import OneHotEncoder


import serial 
import serial.tools.list_ports
import joblib

"""
1. Take in input data. Control the amount of data that comes in (1 data point every 3 seconds).
2. Pre-process the data point. 
3. Run data point through ML Model
4. Output the Prediction 

Install dependencies:

    python -m pip install pyserial
"""
LABELS = ["coffee", "sandalwood","unknown"]
i = 0
testFilePath = os.path.join('output/testing/',os.listdir('output/testing/')[i])


# ###WORKIING###
# mins = np.array([8537.3, 8537.2, 3977.6, 3976.2, 3977.1, 2210.4, 2210.3, 2211.2, 6330.0, 6329.7, 6330.2, 2877.2, 2876.9, 2876.9, 13131.9, 13132.3, 13449.4, 13426.5, 2514.1, 2514.0, 2514.0, 1720.5, 1720.1, 1720.7, 6179.5, 6179.3, 6178.4, 12282.9, 12285.4, 12284.6, 8384.5, 8384.3, 25443.8, 25445.9, 17577.7, 17588.0, 17563.5, 23864.1, 23886.9, 23897.8, 17980.3, 17977.5, 17970.7, 9861.1, 9872.4, 9875.7, 64656.6, 64673.9, 1535636.8, 1497775.8, 23023.9, 23023.8, 23009.9, 6693.5, 6692.8, 6692.7, 1092.5, 1092.5, 1862.1, 1862.7, 1863.4, 14600.0, 14605.5, 26.06, 31.81])

# ranges = np.array([688.8, 687.9, 1515.7, 1513.4, 1513.5, 608.6, 607.9, 607.4, 1536.5, 1538.0, 1537.4, 725.8, 726.6, 726.4, 704.6, 702.7, 955.6, 954.9, 391.0, 390.9, 391.0, 115.3, 115.5, 115.4, 941.2, 940.2, 939.3, 2532.8, 2536.4, 2539.9, 840.4, 841.1, 1745.2, 1985.2, 1750144.8, 1714393.2, 2761.7, 2798.4, 2820.3, 2811.5, 4863.9, 4852.5, 4850.3, 1452.5, 1442.5, 1451.3, 2666.4, 2562.3, 176421.7, 337518.8, 661.6, 649.7, 667.5, 1096.7, 1095.3, 1095.1, 0.1, 0.1, 156.3, 156.5, 156.8, 2354.9, 2355.3, 3.28, 19.03])

mins = np.array([8537.3, 8537.2, 3977.6, 3976.2, 3977.1, 2210.4, 2210.3, 2211.2, 6330.0, 6329.7, 6330.2, 2877.2, 2876.9, 2876.9, 13131.9, 13132.3, 13449.4, 13426.5, 2514.1, 2514.0, 2514.0, 1502.6, 1502.2, 1502.9, 4454.6, 4454.6, 4454.8, 9418.5, 9417.1, 9417.7, 8384.5, 8384.3, 25443.8, 25445.9, 15045.3, 15048.7, 15041.6, 23136.3, 23147.7, 23174.4, 17980.3, 17977.5, 17970.7, 9858.7, 9860.5, 9864.4, 64656.6, 64673.9, 1477884.1, 1485443.4, 18127.2, 18128.5, 18129.8, 6693.5, 6692.8, 6692.7, 1092.5, 1092.5, 1743.4, 1743.9, 1744.6, 14600.0, 14605.5, 22.38, 30.99])

ranges = np.array([8126.1, 8123.6, 22705.4, 22688.2, 22699.5, 40997.3, 40972.3, 41012.4, 21101.0, 21095.1, 21092.0, 18730.3, 18735.4, 18733.5, 8591.0, 8584.0, 14465.4, 14469.0, 7455.1, 7454.2, 7454.5, 12641.5, 12648.1, 12652.1, 39860.1, 39860.1, 39846.7, 58464.9, 58495.7, 58459.9, 14405.7, 14399.4, 18904.2, 18892.6, 1830203.8, 1804261.8, 34346.0, 123724.9, 123695.4, 123379.3, 132172.0, 131629.2, 131728.3, 72434.7, 72530.9, 72527.0, 54750.7, 55119.3, 553752.4, 511775.1, 62480.1, 62525.8, 62516.9, 17831.9, 17814.2, 17803.8, 0.1, 0.1, 7329.9, 7338.0, 7342.3, 20829.4, 20882.1, 8.58, 29.81])

one_hot_encoder = OneHotEncoder()



# loaded_model = [load_model("PoCmodel.h5"), "NN"] #neural network
loaded_model = [joblib.load('logisticRegression.pkl'), "LogReg"] #decision tree
# loaded_model = [joblib.load('GradientBoosted'), "GB"] #gradient boosted
# loaded_model = [joblib.load('DecisionTreeModel.pkl'), "Tree"] #decision tree
# loaded_model = [joblib.load('randomForestModel.pkl'), "RF"] #random forest
# loaded_model =[ joblib.load('knnModel.pkl'), "KNN"] #K-Nearest Neighbors
# loaded_model = [joblib.load('TreeNoTorH.pkl'), "TreeNoT"] #decision tree without temperature or humidity



def normalizeData(mins, ranges, raw_data):
  return (raw_data - mins) / ranges

def preprocessing(raw_data): #data type = np array
  header = len(raw_data)

   #we want it to only check for dropped columns
  PREP_DROP = -1 
  PREP_NONE = 0 
  PREP_STD = 1
  PREP_NORM = 2   


  preproc = [PREP_NORM,   # 1 ch 999
           PREP_NORM,   # 2 ch 999
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
        #   print("Dropping column", i+1)
          continue

      # Otherwise, append the column value to the preprocessed data
      prep_data = np.append(prep_data, raw_data[i])
   
  return normalizeData(mins, ranges, prep_data).reshape(1, -1)

def processing(prepped_data, LABELS):
    predictions = loaded_model[0].predict(prepped_data)

    #Neural Network###

    if loaded_model[1] == "NN" or loaded_model[1] == "KNN":
        print(predictions)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_class_labels = [LABELS[label_index] for label_index in predicted_labels]
        print("Predicted class labels:", predicted_class_labels)
    else:
        print("Predicted Label:", predictions)


# Command line arguments
port = "/dev/cu.usbserial-1110"  # Example port, replace with your actual port
baudrate = 115200  # Example baudrate, replace with your actual baudrate

# Open serial port
try:
    ser = serial.Serial(port, baudrate)
except Exception as e:
    print("Error opening serial port:", e)
    exit()


try:
    while True and i <18:
        # start_time = time.time()  # Record the start time of each iteration

        # Read bytes from serial port until a complete data sample is received
        rx_buf = b''
        while True:
            if ser.in_waiting > 0:
                rx_buf += ser.read()
                if rx_buf[-2:] == b'\r\n':
                    break
        
        # Process the received data (example: decode it)
        received_data = rx_buf.decode('utf-8').strip().replace('\r', '')

        # Split values by comma
        data_values = received_data.split(';')
        data_values = data_values[1:]
        raw_data = []
    


        for value in data_values:
            try:
                raw_data.append(float(value))
            except ValueError:
                pass

        # print(raw_data)
        # print(len(raw_data))
        
        # raw_data = raw_data[1:]
        raw_data = np.array(raw_data).astype(float)

        # testFilePath = os.path.join('output/testing/',os.listdir('output/testing/')[i])
        # print(testFilePath)

        # with open(testFilePath, 'r') as f:
        #     csv_reader = csv.reader(f, delimiter=';')
        #     data = next(csv_reader)[1:]
        # raw_data = np.array(data).astype(float)





        # # Preprocess the raw data
        if loaded_model[1] == "NN" or loaded_model[1] == "KNN":
            prepped_data = one_hot_encoder.transform(preprocessing(raw_data).reshape(1, -1))
        else:
            prepped_data = preprocessing(raw_data)

        # # Process the preprocessed data
        processing(prepped_data, LABELS)

        # i += 1

        # Wait for 2 seconds before collecting the next data sample
        # time.sleep(2)

except KeyboardInterrupt:
    pass  # Exit gracefully if Ctrl+C is pressed

# Close serial port
ser.close()
print()
print("Available serial ports:")
available_ports = serial.tools.list_ports.comports()
for port, desc, hwid in sorted(available_ports):
    print("  {} : {} [{}]".format(port, desc, hwid))