import PySimpleGUI as sg #pip3 install pysimplegui
import csv
import os 
import re
import os
import time
import numpy as np
import keras
from keras.models import load_model
import asyncio

import serial 
import serial.tools.list_ports


LABELS = ["coffee", "orange", "unknown"]

mins = np.array([11832.6, 11830.7, 6478.0, 6479.4, 6479.2, 4892.6, 4891.9, 4892.3, 8149.5, 8148.9, 8148.5, 4512.8, 4512.4, 4512.8, 17366.4, 17366.9, 20199.8, 20162.3, 4134.1, 4133.9, 4133.8, 2206.6, 2206.5, 2207.5, 7679.4, 7675.3, 7676.4, 16481.3, 16485.3, 16491.2, 13152.2, 13148.5, 29304.1, 29300.7, 21153.4, 21152.0, 21130.4, 33166.8, 33225.9, 33236.4, 29919.6, 29869.3, 29866.7, 20853.9, 20852.0, 20867.9, 84672.6, 84737.7, 1631367.4, 1548417.5, 23295.6, 23293.6, 23296.7, 10194.8, 10192.8, 10194.6, 1092.5, 1092.5, 2058.3, 2058.5, 2059.7, 18641.1, 18652.2, 27.77, 30.86])

ranges = np.array([1192.7, 1197.8, 3652.5, 3652.9, 3644.7, 1500.2, 1498.1, 1495.5, 1288.8, 1289.5, 1288.7, 5180.9, 5180.9, 5180.3, 888.6, 896.1, 2257.4, 2256.9, 472.6, 473.1, 474.3, 1015.4, 1015.8, 1015.8, 4150.4, 4160.6, 4152.6, 11670.7, 11634.4, 11651.3, 1546.8, 1548.6, 2090.4, 2117.6, 5048.8, 5082.1, 5104.5, 9802.9, 9841.6, 9810.3, 9384.9, 9406.0, 9381.7, 6160.6, 6154.5, 6134.8, 15379.5, 15533.7, 358508.1, 346760.0, 2363.3, 2380.6, 2388.1, 176.1, 175.0, 172.0, 0.1, 0.1, 214.9, 215.4, 215.2, 1890.2, 1886.8, 0.94, 28.84])


loaded_model = load_model("PoCmodel.h5")

def normalizeData(mins, ranges, raw_data):
  return (raw_data - mins) / ranges

def preprocessing(raw_data): #data type = list 
  raw_data = raw_data[1:]
  raw_data = np.array(raw_data).astype(float)
#   header = len(raw_data)

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
   
  return normalizeData(mins, ranges, prep_data).reshape(1, -1)

def processing(prepped_data, LABELS):
    predictions = loaded_model.predict(prepped_data)
    prob = predictions.max()
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_class_labels = [LABELS[label_index] for label_index in predicted_labels]
    return prob, predictions, predicted_class_labels

probability = 0.845  #random default value
prediction = 'coffee' #random default value
allProb = None

# Command line arguments
port = "/dev/cu.usbserial-110"  # Example port, replace with your actual port
baudrate = 115200  # Example baudrate, replace with your actual baudrate

# testFilePath = os.path.join('output/testing/',os.listdir('output/testing/')[1])


font = ("Arial", 20)
# ----------- Start Layout -----------
startLayout = [
    [
        [
            [sg.Text('Nasal.ai Scent Detector', font=font)],
            [sg.Text('Bring sensor close to object and press \nstart to detect smell. \n \n', font=('Arial', 15), key="-STARTTEXT-", visible=True)],
            [sg.Button('Start', size=(5,1.25), key='-START-'), sg.Button('Cancel', size=(6,1.25), key='-CANCEL-', visible=True)]
        ],

        [
            [sg.Text('Probability: ', font=("Arial", 15), key="-PROBABILITY-", visible=False)],
            [sg.Text('Predicted Smell: ', font=("Arial", 15), key="-PREDICTION-", visible=False)],
            [sg.Button('Stop', key='-STOP-', size=(5,1.25), visible=False)]
        ]
        
    ]
]
window = sg.Window('Nasal.ai scent detector', startLayout)

async def process_data(window):
    rx_buf = b''
    global i
    while START and i <18:
        global probability, prediction
        testFilePath = os.path.join('output/testing/', os.listdir('output/testing/')[i])
        print(testFilePath)

        with open(testFilePath, 'r') as f:
            csv_reader = csv.reader(f, delimiter=';')
            raw_data = next(csv_reader)

        # if ser.in_waiting > 0:
            #     rx_buf += ser.read()
            #     if rx_buf[-2:] == b'\r\n':
            #         break
        
            # # Process the received data (example: decode it)
            # received_data = rx_buf.decode('utf-8').strip().replace('\r', '')
         # #Split values by comma
        # data_values = received_data.split(';')
        # raw_data = []

        # for value in data_values:
        #     try:
        #         raw_data.append(float(value))
        #     except ValueError:
        #         pass 

        
        # Perform data processing and prediction
        preprocessed_data = preprocessing(raw_data)
        probability, allProb, prediction = processing(preprocessed_data, LABELS)
        
        # Update GUI with prediction and probability
        window['-PROBABILITY-'].update("Probability: {}".format(probability))
        window['-PREDICTION-'].update("Prediction: {}".format(prediction))
        
        i += 1 
        await asyncio.sleep(2)  # Wait for 2 seconds before processing next data
        


# try:
#         ser = serial.Serial(port, baudrate)
# except Exception as e:
#         print("Error opening serial port:", e)
#         exit()



async def event_loop(window):
    while True:
        event, values = window.read()
        if  event == sg.WIN_CLOSED or event=='-CANCEL-':
            break
        if event == '-START-': #start collecting, preprocessing, and processing sensor readings
            try:
                global START
                START = True
                window['-STARTTEXT-'].update(visible=False)
                window['-START-'].update(visible=False)
                window['-CANCEL-'].update(visible=False)

                window['-PROBABILITY-'].update("Probability: ",visible=True)
                window['-PREDICTION-'].update("Prediction: ", visible=True)
                window['-STOP-'].update(visible=True)
                
                await asyncio.create_task(process_data(window))
            except:
                pass
        if event == '-STOP-':  #terminate the testing code (no more sensor reading)
            try:     
                window['-STARTTEXT-'].update(visible=True)
                window['-START-'].update(visible=True)
                window['-CANCEL-'].update(visible=True)

                window['-PROBABILITY-'].update(visible=False)
                window['-PREDICTION-'].update(visible=False)
                window['-STOP-'].update(visible=False)      
            except:
                pass
async def main():
    await event_loop(window)

# ser.close()
asyncio.run(main())
window.close()


