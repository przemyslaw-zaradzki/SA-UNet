###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

import argparse
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import (confusion_matrix, f1_score,
                             jaccard_similarity_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from  SA_UNet import *



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=384, required=False)
parser.add_argument('--size', default=384, type=int, required=False)
args = parser.parse_args()


path_experiment = f'Model/{args.dataset}/'
test_images_loc = f'data/{args.dataset}/test/images/'
test_label_loc = f'data/{args.dataset}/test/manual/'

test_files = os.listdir(test_images_loc)
test_data = []
test_label = []

desired_size = args.size

for i in test_files:
    im = Image.open(test_images_loc + i)
    label = Image.open(test_label_loc + i.split('fundus')[0] + 'fundus_ref.bmp').convert('L')

    im = np.array(im)
    label = np.array(label)

    im = im[:desired_size,:desired_size]
    label = label[:desired_size,:desired_size]

    test_data.append(im)
    _, temp = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)

test_data = np.array(test_data)
test_label = np.array(test_label)

x_test = test_data.astype('float32') / 255.
y_test = test_label.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

model=SA_UNet(input_size=(desired_size,desired_size,3),start_neurons=16,lr=1e-3,keep_prob=0.82,block_size=7)
weight=f"Model/{args.dataset}/{args.dataset}_SA_UNet.h5"
model.load_weights(weight)

predictions = model.predict(x_test)

#====== Evaluate the results
print("\n\n========  Evaluate the results =======================")
y_scores = predictions.flatten()
y_true = y_test.flatten()

#Area under the ROC curve
y_true = np.round(y_true).astype('uint8')
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"ROC.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"Precision_recall.png")

#Confusion matrix
threshold_confusion = 0.5
print("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print(confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print("Precision: " +str(precision))

#Jaccard similarity index
jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
print("\nJaccard similarity score: " +str(jaccard_index))

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print("\nF1 score (F-measure): " +str(F1_score))

#Save the results
file_perf = open(path_experiment+'performances.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
file_perf.close()