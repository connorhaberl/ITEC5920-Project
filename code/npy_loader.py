import pandas as pd
import numpy as np
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
#np_file = os.path.join(dir_path,'../output/exp1.1/test_bootstrap_ids.npy')

# truth_labels_fp = os.path.join(dir_path,'../output/exp1.1.1/data/y_test.npy')
# rf_labels_fp = os.path.join(dir_path,'../output/exp1.1.1/models/random_forest/y_test_pred.npy')
# resnet_labels_fp = os.path.join(dir_path,'../output/exp1.1.1/models/fastai_xresnet1d101/y_test_pred.npy')
exp = "exp3"
#cascade classifier
truth_labels_fp = os.path.join(dir_path,'../output/latest2/cascade/'+exp+'/data/y_test.npy')
cascade_resnet_labels_fp = os.path.join(dir_path,'../output/latest2/cascade/'+exp+'/models/cascade_classifier_resnet/y_test_pred.npy')
cascade_labels_fp = os.path.join(dir_path,'../output/latest2/cascade/'+exp+'/models/cascade_classifier/y_test_pred.npy')

truth_labels = pd.DataFrame(np.load(truth_labels_fp, allow_pickle=True))
cascade_resnet = pd.DataFrame(np.load(cascade_resnet_labels_fp, allow_pickle=True))
cascade = pd.DataFrame(np.load(cascade_labels_fp, allow_pickle=True))

truth_labels.to_csv('../output/latest2/y_test_true_'+exp+'.csv')
cascade_resnet.to_csv('../output/latest2/y_test_cascade_resnet_pred_'+exp+'.csv')
cascade.to_csv('../output/latest2/y_test_cascade_pred_'+exp+'.csv')
#data = np.load('../output/exp0/val_bootstrap_ids.npy', allow_pickle=True)

# print(truth_labels.head(5))
# print(rf_labels.head(5))
# print(resnet_labels.head(5))
# print(truth_labels.shape)

# print(df2.head(100))
# print(df2.shape)