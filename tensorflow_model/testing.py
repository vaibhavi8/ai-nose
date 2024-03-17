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
LABELS = ["coffee", "sandalwood", "orange", "unknown"]
i = 0
testFilePath = os.path.join('output/testing/',os.listdir('output/testing/')[i])



mins = np.array([11275.8, 11275.0, 8810.7, 8805.7, 8809.4, 4527.9, 4528.2, 4530.1, 10143.3, 10143.4, 10143.7, 6015.4, 6014.5, 6013.1, 16845.0, 16852.6, 19108.7, 19080.7, 4020.6, 4020.3, 4020.6, 2231.7, 2231.6, 2232.2, 7562.9, 7561.7, 7559.8, 18375.6, 18377.3, 18382.4, 11662.7, 11661.7, 29763.8, 29763.2, 20690.7, 20691.8, 20711.9, 35375.4, 35391.5, 35432.9, 36025.6, 35999.3, 36019.0, 17350.1, 17359.8, 17360.0, 83266.5, 83215.2, 1533263.9, 1565930.4, 23046.8, 23030.6, 23040.8, 9630.8, 9631.2, 9630.1, 1092.5, 1092.5, 2018.5, 2019.2, 2020.0, 19330.1, 19334.3, 27.32, 34.92])

ranges = np.array([3209.3, 3210.4, 9959.1, 9970.8, 9963.9, 12560.6, 12560.5, 12558.7, 5557.6, 5555.9, 5556.9, 15592.1, 15597.8, 15597.3, 3127.7, 3116.4, 5828.6, 5853.3, 3306.3, 3306.6, 3306.8, 5392.9, 5394.6, 5396.2, 18437.7, 18435.0, 18430.5, 48666.5, 48649.3, 48594.0, 6109.0, 6103.1, 9123.3, 9161.5, 28700.6, 28724.8, 28675.7, 64109.4, 64103.3, 64123.5, 87784.0, 87747.3, 87640.1, 64943.3, 65031.6, 65031.4, 15573.6, 15139.5, 498372.6, 457764.5, 24127.9, 24146.9, 24173.8, 8013.3, 8010.2, 8030.6, 0.1, 0.1, 2913.9, 2915.9, 2917.6, 10602.0, 10603.4, 3.64, 25.88])

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

def processing(prepped_data, LABELS):
    predictions = loaded_model.predict(prepped_data)
    print(predictions)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_class_labels = [LABELS[label_index] for label_index in predicted_labels]
    print("Predicted class labels:", predicted_class_labels)


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