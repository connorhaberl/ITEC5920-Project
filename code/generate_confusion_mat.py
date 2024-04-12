import pandas as pd
import numpy as np
import os
import sklearn.metrics as metrics
import dill as pickle
from matplotlib import pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

#set the model name and experiment
model = 'cascade_classifier_resnet'
experiment = 'exp3'

if(experiment=='exp1.1.1'): 
    base_fp = os.path.join(dir_path,'../output/' + experiment)
else:
    base_fp = os.path.join(dir_path,'../output/latest2/cascade/' + experiment)


#For experiment 1.1, 2 or 3 cascade classifiers



actual_file = os.path.join(base_fp, 'data/y_test.npy')
predicted_file = os.path.join(base_fp, 'models/' + model + '/y_test_pred.npy')

class_labels_fp = os.path.join(base_fp, 'data/mlb.pkl') ##Need to figure out the file type & name here
mlb = pickle.load(open(class_labels_fp, 'rb'))

class_labels = list(mlb.classes_)

actual = np.load(actual_file, allow_pickle=True)
predicted = np.load(predicted_file, allow_pickle=True)

confusion_matrix = metrics.confusion_matrix(actual.argmax(axis=1), predicted.argmax(axis=1), labels = range(len(class_labels)))

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels= range(len(class_labels)))

cm_display.plot()
plt.show()
plt.savefig('../outputs/confusion_matrix_' + model+'_'+ experiment + '.png')
