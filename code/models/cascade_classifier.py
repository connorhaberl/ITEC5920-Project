from models.base_model import ClassificationModel
from models.fastai_model import fastai_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import torch
import dill as pickle


def get_dimensions(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_dimensions(lst[0])
    else:
        return []

def transform(x, chunk_size=1000):
    num_signals = x.shape[0]
    all_signals = np.zeros((num_signals, 1000*12))

    for i in range(num_signals):
        concatenated_data = x[i].flatten()
        all_signals[i] = concatenated_data
        # for j in range(0, len(x[i]), chunk_size):
        #     chunk = np.concatenate(x[i][j:j+chunk_size], axis=0)
        #     concatenated_data.append(chunk)
    
    return all_signals



class cascade_classifier(ClassificationModel):
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape):
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape
        self.model = RandomForestClassifier(n_estimators=10, n_jobs=1)
        # May need to initialize the models here for each superclass model


    def fit(self, X_train, y_train, X_val, y_val, experiment_name):
        #Load pre-trained models

        #Load SKLearn Model for RF
        rf_fp = os.path.join(self.outputfolder, '../../../../../exp1.1.1/models/random_forest/models/random_forest_model.pkl') ##Need to figure out the file type & name here
        with open(rf_fp, 'rb') as f:
            self.model = pickle.load(f)
        #Load Pytorch Models for other classifiers
        # Requires each model saved in a folder nested one deeper than the normal output folder. If using a "output/something/" for the main output, need to add a ../ to each path

        
        #norm_fp = os.path.join('../output', 'NORM_only', experiment_name,'/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pth') #Can't get this to work with os path
        
        #state = torch.load(norm_fp)

        # if experiment_name == 'exp1.1':
        #     n_classes_pretrained = 23
        #     num_classes = 23
        # elif experiment_name == 'exp2':
        #     n_classes_pretrained = 19
        #     num_classes = 19
        # elif experiment_name == 'exp3':
        #     n_classes_pretrained = 12
        #     num_classes = 12
        
        # norm_fp = self.outputfolder + '../../../NORM_only/' + experiment_name + '/models/fastai_xresnet1d101/models/'
        # CD_fp = self.outputfolder + '../../../CD_only/' + experiment_name + '/models/fastai_xresnet1d101/models/'
        # MI_fp = self.outputfolder + '../../../MI_only/' + experiment_name + '/models/fastai_xresnet1d101/models/'
        # HYP_fp = self.outputfolder + '../../../HYP_only/' + experiment_name + '/models/fastai_xresnet1d101/models/'
        # STTC_fp = self.outputfolder + '../../../STTC_only/' + experiment_name + '/models/fastai_xresnet1d101/models/'

        # norm_mpath = self.outputfolder + '../../../NORM_only/' + experiment_name + '/models/fastai_xresnet1d101/'

        # self.model_CD = torch.load(CD_fp)
        # self.model_MI = torch.load(MI_fp)
        # self.model_HYP = torch.load(HYP_fp)
        # self.model_STTC = torch.load(STTC_fp)

        sttc_fp = os.path.join(self.outputfolder, '../../../../STTC/' + experiment_name + '/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pkl') ##Need to figure out the file type & name here
        with open(sttc_fp, 'rb') as f:
            self.model_STTC = pickle.load(f)

        norm_fp = os.path.join(self.outputfolder, '../../../../NORM/' + experiment_name + '/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pkl') ##Need to figure out the file type & name here
        with open(norm_fp, 'rb') as f:
            self.model_NORM = pickle.load(f)

        cd_fp = os.path.join(self.outputfolder, '../../../../CD/' + experiment_name + '/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pkl')##Need to figure out the file type & name here
        with open(cd_fp, 'rb') as f:
            self.model_CD = pickle.load(f)

        mi_fp = os.path.join(self.outputfolder, '../../../../MI/' + experiment_name + '/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pkl')##Need to figure out the file type & name here
        with open(mi_fp, 'rb') as f:
            self.model_MI = pickle.load(f)

        hyp_fp = os.path.join(self.outputfolder,'../../../../HYP/' + experiment_name + '/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pkl') ##Need to figure out the file type & name here
        with open(hyp_fp, 'rb') as f:
            self.model_HYP = pickle.load(f)

        self.experiment = experiment_name
        pass
    

    def predict(self, X):
        #Use cascade classifier
        mlb_fp = os.path.join(self.outputfolder, '../../../../../exp1.1.1/data/mlb.pkl')
        mlb = pickle.load(open(mlb_fp, 'rb'))
        X_t = transform(X)
        rf_predictions = np.array(self.model.predict(X_t))
        
        norm_col = 3 # TO DO: Could set programatically using the mlb 
        single_col=[0]*rf_predictions.shape[0]

        #format RF predictions to list index for each row
        for i in range(rf_predictions.shape[0]):
            seen_it=False
            for j in range(5):
                if rf_predictions[i][j]==1:
                    if seen_it:
                        #Seen it twice
                        single_col[i] = norm_col
                    else:
                        single_col[i] = j
                        seen_it=True
                
            if seen_it==False:
                single_col[i] = norm_col
       
        NORM_predictions = np.array(self.model_NORM.predict(X))
        CD_predictions = np.array(self.model_CD.predict(X))
        MI_predictions = np.array(self.model_MI.predict(X))
        HYP_predictions = np.array(self.model_HYP.predict(X))
        STTC_predictions = np.array(self.model_STTC.predict(X))


        

        #print("RF Pred Dimensions",get_dimensions(rf_predictions))
        #print("Norm Pred Dimensions",get_dimensions(STTC_predictions))
        print("RF Pred Dimensions:" + str(rf_predictions.shape))
        print("Norm Pred Dimensions"+ str(NORM_predictions.shape))


        # #Create a list of all the predictions from each model
        second_tier_predictions = {"CD":CD_predictions, "HYP":HYP_predictions,  "MI":MI_predictions, "NORM":NORM_predictions, "STTC":STTC_predictions}
        
        all_keys_fp = os.path.join(self.outputfolder, '../../../../../' + self.experiment + '/data/mlb.pkl')
        print(all_keys_fp)
        all_mlb = pickle.load(open(all_keys_fp, 'rb'))
        all_classes = list(all_mlb.classes_)
        print(f"ac: {all_classes}")

        for superclass in second_tier_predictions.keys():
            
            mlb_fp = os.path.join(self.outputfolder, '../../../../' + superclass + '/' + self.experiment + '/data/mlb.pkl')
            mlb_partial = pickle.load(open(mlb_fp, 'rb'))
            partial_classes = list(mlb_partial.classes_)
            print(partial_classes)
            partial_data = second_tier_predictions[superclass]

            rows,cols = partial_data.shape

            for index, key in enumerate(all_classes):
                if key not in partial_classes:
                    partial_data = np.hstack((partial_data[:,:index],np.zeros((rows,1)),partial_data[:,index:]))

            second_tier_predictions[superclass] = partial_data

        # #Select only the predictions from each model based on RF predictions
        
        superclass_lkp = list(mlb.classes_)
        
        #print("superclass incices: ", enumerate(superclass_indices))
        #print("superclass len: ", len(enumerate(superclass_indices)))
        print("predictions shape CD" , second_tier_predictions['CD'].shape)
        print("predictions shape HYP" , second_tier_predictions['HYP'].shape)
        print("predictions shape MI" , second_tier_predictions['MI'].shape)
        print("predictions shape NORM" , second_tier_predictions['NORM'].shape)
        print("predictions shape STTC" , second_tier_predictions['STTC'].shape)
        print("superclass lkp", superclass_lkp)
        print("\n\ninput shape", X.shape)

        predictions = np.zeros(second_tier_predictions['CD'].shape)
        
        for row, sc_index in enumerate(single_col):          
            predictions[row,:] = second_tier_predictions[superclass_lkp[sc_index]][row,:]
            #predictions.append(second_tier_predictions[superclass_lkp[sc_index]][row])
        
        #predictions = np.array(predictions)
        print(f'Predictions: {predictions.shape}')
        #print(predictions[:5,5:20])
        #print(predictions[:10,:])

        return predictions
        pass

