import PySimpleGUI as sg #pip3 install pysimplegui
import os 
import re
import time

probability = 0.845  #random default value
prediction = 'coffee' #random default value


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

while True:
    event, values = window.read()
    if  event == sg.WIN_CLOSED or event=='-CANCEL-':
        break
    if event == '-START-': #start collecting, preprocessing, and processing sensor readings
        try:
            window['-STARTTEXT-'].update(visible=False)
            window['-START-'].update(visible=False)
            window['-CANCEL-'].update(visible=False)

            window['-PROBABILITY-'].update("Probability: {}".format(probability),visible=True)
            window['-PREDICTION-'].update("Prediction: {}".format(prediction), visible=True)
            window['-STOP-'].update(visible=True)
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

window.close()

