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

"""
1. Take in input data. Control the amount of data that comes in (1 data point every 3 seconds).
2. Pre-process the data point. 
3. Run data point through ML Model
4. Output the Prediction 

Install dependencies:

    python -m pip install pyserial
"""
# class_labels = ['coffee', 'irishcream', 'kahlua', 'rum', 'test'] 
i = 0
testFilePath = os.path.join('output/testing/',os.listdir('output/testing/')[i])

class_labels = ['coffee', 'orange', 'unknown'] 

mins = np.array([11832.6, 11830.7, 6478.0, 6479.4, 6479.2, 4892.6, 4891.9, 4892.3, 8149.5, 8148.9, 8148.5, 4512.8, 4512.4, 4512.8, 17366.4, 17366.9, 20199.8, 20162.3, 4134.1, 4133.9, 4133.8, 2206.6, 2206.5, 2207.5, 7679.4, 7675.3, 7676.4, 16481.3, 16485.3, 16491.2, 13152.2, 13148.5, 29304.1, 29300.7, 21153.4, 21152.0, 21130.4, 33166.8, 33225.9, 33236.4, 29919.6, 29869.3, 29866.7, 20853.9, 20852.0, 20867.9, 84672.6, 84737.7, 1631367.4, 1548417.5, 23295.6, 23293.6, 23296.7, 10194.8, 10192.8, 10194.6, 1092.5, 1092.5, 2058.3, 2058.5, 2059.7, 18641.1, 18652.2, 27.77, 30.86])

ranges = np.array([1192.7, 1197.8, 3652.5, 3652.9, 3644.7, 1500.2, 1498.1, 1495.5, 1288.8, 1289.5, 1288.7, 5180.9, 5180.9, 5180.3, 888.6, 896.1, 2257.4, 2256.9, 472.6, 473.1, 474.3, 1015.4, 1015.8, 1015.8, 4150.4, 4160.6, 4152.6, 11670.7, 11634.4, 11651.3, 1546.8, 1548.6, 2090.4, 2117.6, 5048.8, 5082.1, 5104.5, 9802.9, 9841.6, 9810.3, 9384.9, 9406.0, 9381.7, 6160.6, 6154.5, 6134.8, 15379.5, 15533.7, 358508.1, 346760.0, 2363.3, 2380.6, 2388.1, 176.1, 175.0, 172.0, 0.1, 0.1, 214.9, 215.4, 215.2, 1890.2, 1886.8, 0.94, 28.84])

loaded_model = load_model("PoCmodel.h5")

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

def processing(prepped_data, class_labels):
    predictions = loaded_model.predict(prepped_data)
    print(predictions)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_class_labels = [class_labels[label_index] for label_index in predicted_labels]
    print("Predicted class labels:", predicted_class_labels)


# Command line arguments
port = "/dev/cu.usbserial-110"  # Example port, replace with your actual port
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
        raw_data = []


        for value in data_values:
            try:
                raw_data.append(float(value))
            except ValueError:
                pass
        raw_data = raw_data[1:]
        raw_data = np.array(raw_data).astype(float)

        # testFilePath = os.path.join('output/testing/',os.listdir('output/testing/')[i])
        # print(testFilePath)

        # with open(testFilePath, 'r') as f:
        #     csv_reader = csv.reader(f, delimiter=';')
        #     data = next(csv_reader)[1:]
        # raw_data = np.array(data).astype(float)




        # Preprocess the raw data
        prepped_data = preprocessing(raw_data)

        # Process the preprocessed data
        processing(prepped_data, class_labels)

        i += 1

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