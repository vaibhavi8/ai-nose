#/usr/bin/env python3
import csv
import argparse
import os
import time
import numpy as np
import keras
from keras.models import load_model

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

mins = np.array([13654.6, 13640.2, 13175.2, 13179.1, 13170.4, 15486.5, 15487.1, 15497.3, 15587.8, 15582.0, 15586.1, 8329.7, 8329.8, 8331.2, 18615.6, 18611.4, 22576.7, 22558.6, 6305.0, 6304.7, 6304.6, 4522.6, 4523.2, 4523.2, 15074.7, 15072.9, 15073.1, 29876.3, 29883.6, 29886.9, 17184.9, 17166.1, 36167.3, 36215.6, 1514926.5, 1457085.9, 31652.4, 73704.2, 73799.8, 73690.0, 64843.5, 64788.8, 64851.0, 38597.3, 38634.8, 38674.2, 96352.6, 96232.6, 1525211.2, 1491958.1, 41202.4, 41238.8, 41268.0, 15972.9, 15972.2, 15970.2, 1092.5, 1092.5, 4357.9, 4360.7, 4362.5, 27946.4, 27951.8, 24.22, 36.52])

ranges = np.array([3008.8, 3020.6, 13507.8, 13485.3, 13506.2, 27721.2, 27695.5, 27726.3, 11843.2, 11842.8, 11836.1, 5562.0, 5560.5, 5564.1, 3107.3, 3104.9, 5338.1, 5336.9, 3664.2, 3663.5, 3663.9, 9621.5, 9627.1, 9631.8, 29240.0, 29241.8, 29228.4, 38007.1, 38029.2, 37990.7, 5605.3, 5617.6, 8180.7, 8122.9, 330322.6, 362224.6, 13541.2, 73157.0, 73043.3, 72863.7, 85308.8, 84817.9, 84848.0, 35327.3, 35333.0, 35308.8, 23054.7, 23560.6, 323475.8, 332369.8, 39404.9, 39415.5, 39378.7, 8552.5, 8534.8, 8526.3, 0.1, 0.1, 4715.4, 4721.2, 4724.4, 7483.0, 7535.8, 5.64, 17.08])


loaded_model = [load_model("PoCmodel.h5"), "NN"] #neural network
loaded_model = [joblib.load('DecisionTreeModel.pkl'), "Tree"] #decision tree
loaded_model = [joblib.load('randomForestModel.pkl'), "RF"] #random forest
loaded_model =[ joblib.load('knnModel.pkl'), "KNN"] #K-Nearest Neighbors
loaded_model = [joblib.load('TreeNoTorH.pkl'), "TreeNoT"] #decision tree without temperature or humidity



def normalizeData(mins, ranges, raw_data):
  return (raw_data - mins) / ranges

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