import csv
import os
import shutil
import keras
import numpy as np
import matplotlib.pyplot as plt
import splitfolders
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from sklearn import metrics
import random

# # # reading the data
# ### Settings
HOME_PATH = ".."              # Location of the working directory
DATASET_PATH = "../dataset"   # Upload your .csv samples to this directory
OUT_PATH = "../out"           # Where output files go (will be deleted and recreated)
OUT_ZIP = "../out.zip"        # Where to store the zipped output files
NANODATA_PATH = "../nano_data"    # Where one-line nano data is stored

# Do not change these settings!
PREP_DROP = -1                      # Drop a column
PREP_NONE = 0                       # Perform no preprocessing on column of data
PREP_STD = 1                        # Perform standardization on column of data
PREP_NORM = 2                       # Perform normalization on column of data

### Combinging .scsv files to create one csv file with all the data.
fields = []
#create a header
header = list(range(1, 65))
header += ["temperature", "humidity"]

rum = 0
rumIter = 0
test = 0
testIter = 0
kahlua = 0
kahluaIter = 0
irishCream = 0
irishCreamIter= 0
coffee = 0
coffeeIter = 0
def writeToFile(filepath1, filepath2): #method that reads from current file, and writes to the appropriate file
  with open(filepath1, 'r') as file1:
    first_line = file1.readline().strip()    #reading the first line
    fields = first_line.split(';')[1:]       #fields is the data split by ';', it skips the first column ('start')
  with open(filepath2, "a") as file2:        #creating/opening the file specified to write the data
    writer = csv.writer(file2)
    if os.path.getsize(os.path.join(DATASET_PATH, filepath2)) == 0:
      writer.writerow(header)                #if header doesn't exist, creating a header
    writer.writerow(fields)                  #inserting all fields in first_line under the appropriate header

### Delete output directory (if it exists) and recreate it
if os.path.exists(DATASET_PATH):
  shutil.rmtree(DATASET_PATH)
os.makedirs(DATASET_PATH)

os.mkdir(os.path.join(DATASET_PATH, "rum"))
os.mkdir(os.path.join(DATASET_PATH, "test"))
os.mkdir(os.path.join(DATASET_PATH, "kahlua"))
os.mkdir(os.path.join(DATASET_PATH, "irishCream"))
os.mkdir(os.path.join(DATASET_PATH, "coffee"))
for filename in os.listdir(NANODATA_PATH):   #going through all the files in nano_data
  filepath = os.path.join(NANODATA_PATH, filename)
  if not os.path.isfile(filepath):
    continue
  if(filename[0:3]=="rum"):                  #checking the label to pipeline the data to the accurate file
    writeToFile(filepath, (os.path.join(DATASET_PATH, f"rum/rum_{rumIter}.csv")))
    rum += 1
    if rum%5 == 0:
      rumIter += 1
  elif(filename[0:4]=="test"):
    writeToFile(filepath, (os.path.join(DATASET_PATH, f"test/test_{testIter}.csv")))
    test += 1
    if test%5 == 0:
      testIter += 1
  elif(filename[0:4]=="kahl"):
    writeToFile(filepath, (os.path.join(DATASET_PATH, f"kahlua/kahlua_{kahluaIter}.csv")))
    kahlua += 1
    if kahlua%5 == 0:
      kahluaIter += 1
  elif(filename[0:4]=="Iris"):
    writeToFile(filepath, (os.path.join(DATASET_PATH, f"irishCream/irishCream_{irishCreamIter}.csv")))
    irishCream += 1
    if irishCream%5 == 0:
      irishCreamIter += 1
  elif(filename[0:4]=="coff"):
    writeToFile(filepath, (os.path.join(DATASET_PATH, f"coffee/coffee_{coffeeIter}.csv")))
    coffee += 1
    if coffee%5 == 0:
      coffeeIter += 1
  else:
    print("Error: category for " + filename + " not found.")

print("Coffee: " + str(coffee))
print("Test: " + str(test))
print("Rum: " + str(rum))
print("Irish Cream: " + str(irishCream))
print("Kahlua: " + str(kahlua))

### Read in .csv files to construct one long multi-axis, time series data

# Store header, raw data, and number of lines found in each .csv file
header = None
raw_data = []
num_lines = []
filenames = []

# Read each CSV file
for dir in os.listdir(DATASET_PATH):
  dirpath = os.path.join(DATASET_PATH, dir)
  for filename in os.listdir(dirpath):

    # Check if the path is a file
    filepath = os.path.join(dirpath, filename)
    filedir = os.path.join(dir, filename)
    if not os.path.isfile(filepath):
      continue

    # Read the .csv file
    with open(filepath) as f:
      csv_reader = csv.reader(f, delimiter=',')

      # Read each line
      valid_line_counter = 0
      for line_count, line in enumerate(csv_reader):

        # Check header
        if line_count == 0:

          # Record first header as our official header for all the data
          if header == None:
            header = line

          # Check to make sure subsequent headers match the original header
          if header == line:
            num_lines.append(0)
            filenames.append(filedir)
          else:
            print("Error: Headers do not match. Skipping", filename)
            break

        # Construct raw data array, make sure number of elements match number of header labels
        else:
          if len(line) == len(header):
            raw_data.append(line)
            num_lines[-1] += 1
          else:
            print("Error: Data length does not match header length. Skipping line.")
            continue

# Convert our raw data into a numpy array
raw_data = np.array(raw_data).astype(float)

# Print out our results
print("Dataset array shape:", raw_data.shape)
print("Number of elements in num_lines:", len(num_lines))
print("Number of filenames:", len(filenames))
assert(len(num_lines) == len(filenames))


### Analyze the data

# Calculate means, standard deviations, and ranges
means = np.mean(raw_data, axis=0)
std_devs = np.std(raw_data, axis=0)
maxes = np.max(raw_data, axis=0)
mins = np.min(raw_data, axis=0)
ranges = np.ptp(raw_data, axis=0)

# Print results
for i, name in enumerate(header):
  print(name)
  print("  mean:", means[i])
  print("  std dev:", std_devs[i])
  print("  max:", maxes[i])
  print("  min:", mins[i])
  print("  range:", ranges[i])

### Choose preprocessing method for each column
#     PREP_DROP: Drop column
#     PREP_NONE: no preprocessing
#     PREP_STD: standardization (if data is Gaussian)
#     PREP_NORM: normalization (if data is non-Gaussian)

# Change this to match your picks!
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

# Check to make sure we have the correct number of preprocessing request elements
assert(len(preproc) == len(header))
assert(len(preproc) == raw_data.shape[1])

# ### If we do not need the timestamp column, drop it from the data
# if not KEEP_TIMESTAMP:
#   header = header[1:]
#   raw_data = raw_data[:,1:]
#   print("Array shape without timestamp:", data_without_time.shape)

### Perform preprocessing steps as requested

# Figure out how many columns we plan to keep
num_cols = sum(1 for x in preproc if x != PREP_DROP)

# Create empty numpy array and header for preprocessed data
prep_data = np.zeros((raw_data.shape[0], num_cols))
prep_header = []
prep_means = []
prep_std_devs = []
prep_mins = []
prep_ranges = []

# Go through each column to preprocess the data
prep_c = 0
for raw_c in range(len(header)):

  # Drop column if requested
  if preproc[raw_c] == PREP_DROP:
    print("Dropping", header[raw_c])
    continue

  # Perform data standardization
  if preproc[raw_c] == PREP_STD:
    prep_data[:, prep_c] = (raw_data[:, raw_c] - means[raw_c]) / std_devs[raw_c]

  # Perform data normalization
  elif preproc[raw_c] == PREP_NORM:
    prep_data[:, prep_c] = (raw_data[:, raw_c] - mins[raw_c]) / ranges[raw_c]

  # Copy data over if no preprocessing is requested
  elif preproc[raw_c] == PREP_NONE:
    prep_data[:, raw_c] = raw_data[:, raw_c]

  # Error if code not recognized
  else:
    raise Exception("Preprocessing code not recognized")

  # Copy header (and preprocessing constants) and increment preprocessing column index
  prep_header.append(header[raw_c])
  prep_means.append(means[raw_c])
  prep_std_devs.append(std_devs[raw_c])
  prep_mins.append(mins[raw_c])
  prep_ranges.append(ranges[raw_c])
  prep_c += 1

# Show new data header and shape
print(prep_header)
print("New data shape:", prep_data.shape)
print("Means:", [float("{:.4f}".format(x)) for x in prep_means])
print("Std devs:", [float("{:.4f}".format(x)) for x in prep_std_devs])
print("Mins:", [float("{:.4f}".format(x)) for x in prep_mins])
print("Ranges:", [float("{:.4f}".format(x)) for x in prep_ranges])

### Delete output directory (if it exists) and recreate it
if os.path.exists(OUT_PATH):
  shutil.rmtree(OUT_PATH)
os.makedirs(OUT_PATH)


### Write out data to .csv files
os.mkdir(os.path.join(OUT_PATH, "rum"))
os.mkdir(os.path.join(OUT_PATH, "test"))
os.mkdir(os.path.join(OUT_PATH, "kahlua"))
os.mkdir(os.path.join(OUT_PATH, "irishCream"))
os.mkdir(os.path.join(OUT_PATH, "coffee"))

# Go through all the original filenames
row_index = 0
for file_num, filename in enumerate(filenames):

  # Open .csv file
  file_path = os.path.join(OUT_PATH, filename)
  with open(file_path, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')

    # Write header
    csv_writer.writerow(prep_header)

    # Write contents
    for _ in range(num_lines[file_num]):
      csv_writer.writerow(prep_data[row_index])
      row_index += 1
      
def split_data(input_folder, output_folder, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    input_data = []
    for dir in os.listdir(input_folder):
      dirPath = os.path.join(input_folder, dir)
      temp = []
      for filename in os.listdir(dirPath):
        temp.append(os.path.join(dir, filename))
      input_data.append(temp)
    
    # Create output folder if it doesn't exist and deleting it if it doesn exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Create train, val, and test folders
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'testing')

    os.mkdir(train_folder)
    os.mkdir(val_folder)
    os.mkdir(test_folder)

    for dir in os.listdir(output_folder):
      os.makedirs(os.path.join(output_folder, dir, "rum"))
      os.makedirs(os.path.join(output_folder, dir, "test"))
      os.makedirs(os.path.join(output_folder, dir, "kahlua"))
      os.makedirs(os.path.join(output_folder, dir, "irishCream"))
      os.makedirs(os.path.join(output_folder, dir, "coffee"))

    # Shuffle the input data
    for i in input_data:
      random.shuffle(i)
      num_samples = len(i)
      num_train = int(train_ratio * num_samples)
      num_val = int(val_ratio * num_samples)
      # num_test = int(test_ratio * num_samples)

      train_data= i[:num_train]
      val_data=i[num_train:num_train + num_val]
      test_data=i[num_train + num_val:]

      copy_files(train_data, train_folder, input_folder)
      copy_files(val_data, val_folder, input_folder)
      copy_files(test_data, test_folder, input_folder)

def copy_files(data, destination_folder, input_folder):
    for i in data:
        filepath = os.path.join(input_folder,i)
        shutil.copy(filepath, os.path.join(destination_folder, i))

split_data('../out', 'output')