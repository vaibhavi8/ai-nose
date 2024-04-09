import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import joblib


from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import csv

import shap

shap.initjs()

LABELS = ["coffee", "sandalwood", "unknown"]
# LABELS = ["coffee", "kahlua", "IrishCream", "rum", "test"]


outputPath = '../out/'
def flatten(path):
  raw_data = []
  features_dirs = []
  data = []
  labels = []
  for dir in os.listdir(path):
    features_dirs.append(dir)
    for file in os.listdir(os.path.join(path, dir)):
      filepath = os.path.join(path, dir, file)
      with open(filepath, newline='') as afile:
        csvreader = csv.reader(afile)
        features = next(csvreader)
        for row in csvreader:
          labels.append(dir)
          data.append(row)
  data = np.array(data).astype(float)
  labels = np.array(labels)

  return data, labels, features

X, y, features = flatten(outputPath)
# X_valid, y_valid, ig= flatten(validPath)

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y)
# y_valid_encoded = label_encoder.transform(y_valid)

df_train = pd.DataFrame(X, columns=features )
df_train['label'] = y_train_encoded

x = df_train.drop('label', axis=1)
y = df_train['label']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.1, stratify=y)


# #multilinear model
# log_reg = LogisticRegression(solver='saga', multi_class='multinomial', penalty="l1", max_iter=5000) #utilizes crossentropy_loss or log_loss
# log_reg.fit(trainX, trainY)
# y_pred = log_reg.predict(testX)
# # log_reg.fit(trainX, trainY)
# # y_pred = log_reg.predict(testX)

# joblib.dump(log_reg, "logisticRegression.pkl")

# coefficients = log_reg.coef_
# feature_names = trainX.columns

# df_coefficients = pd.DataFrame({'Feature': feature_names, 'Coefficient': np.abs(coefficients[0])})

# df_coefficients = df_coefficients.sort_values(by='Coefficient', ascending=False)
# print(df_coefficients.head(28))

# print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
# print('Error rate: {:.2f}'.format(1 - accuracy_score(testY, y_pred)))

# # clf = LogisticRegression(solver='newton-cg', multi_class='multinomial')
# scores = cross_val_score(log_reg, trainX, trainY, cv=5)
# scores #printing out the scores
# print("scores: ", scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# confusion_matrix = confusion_matrix(testY, y_pred)
# print(confusion_matrix)


# # plt.matshow(confusion_matrix, cmap=plt.cm.gray)
# # plt.show()


# # probability = log_reg.predict_proba(testX)
# # print(probability)

# df_results = pd.DataFrame(log_reg.predict_proba(testX), columns=log_reg.classes_)
# df_results['predicted_classes'] = y_pred
# df_results['actual_class'] = testY.to_frame().reset_index().drop(columns='index')

# df_incorrect = df_results[df_results['predicted_classes'] != df_results['actual_class']]

# print(df_incorrect.head(28))
# # print(df_results.head())


svm_model = SVC(kernel='linear', decision_function_shape='ovo', C=0.65) #ovo is computationally expensive for a large number of classes and ovr can suffer issues with class imbalance, right now because we have few classes, we use ovo. also, linear is giving best accuracy right now. 

svm_model.fit(trainX, trainY)

y_train_pred = svm_model.predict(trainX)
y_valid_pred = svm_model.predict(testX)
# Calculate training accuracy
train_accuracy = accuracy_score(trainY, y_train_pred)
val_accuracy = accuracy_score(testY, y_valid_pred)
print("SVM Training Accuracy:", train_accuracy)
print("SVM Validation Accuracy:", val_accuracy)


lin_svm = LinearSVC(penalty='l2', max_iter=100000, dual='auto', C=0.85, loss='squared_hinge')
lin_svm.fit(trainX, trainY)

y_train_pred = lin_svm.predict(trainX)
y_valid_pred = lin_svm.predict(testX)
# Calculate training accuracy
train_accuracy = accuracy_score(trainY, y_train_pred)
val_accuracy = accuracy_score(testY, y_valid_pred)
print("Linear SVM Training Accuracy:", train_accuracy)
print("Linear SVM Validation Accuracy:", val_accuracy)

joblib.dump(lin_svm, 'lin_svm.pkl')

guassian_model = MultinomialNB()
guassian_model.fit(trainX, trainY)

train_pred = guassian_model.predict(trainX)
val_pred = guassian_model.predict(testX)

train_accuracy = accuracy_score(trainY, train_pred)
val_accuracy = accuracy_score(testY, val_pred)

print("Naive Bayes Training Accuracy:", train_accuracy)
print("Naive Bayes Validation Accuracy:", val_accuracy)

joblib.dump(guassian_model, 'naiveBayes.pkl')