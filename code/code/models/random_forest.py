from models.base_model import ClassificationModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os


#prepare data for random forest
def transform(x):
    num_signals = x.shape[0]
    all_signals = np.zeros((num_signals, 1000*12))

    for i in range(num_signals):
        concatenated_data = x[i].flatten()
        all_signals[i] = concatenated_data
        
    
    return all_signals



class random_forest(ClassificationModel):
    #initialize model
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape):
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape
        self.model = RandomForestClassifier(n_estimators=10, n_jobs=1)
    #train model
    def fit(self, X_train, y_train, X_val, y_val):
        self.model  = RandomForestClassifier(n_estimators=1000, n_jobs=100,verbose = 100)

        X_train_t = transform(X_train)
        X_val_v = transform(X_val)

        self.model.fit(X_train_t, y_train)
     
        os.makedirs(os.path.join(self.outputfolder, 'models/'), exist_ok=True)

        # save
        model_fp = os.path.join(self.outputfolder, 'models/random_forest_model.pkl') #Should be changed to save for different experiments
        with open(model_fp,'wb+') as f:
            pickle.dump(self.model,f)
        pass
    
    #make a prediction
    def predict(self, X):
        X_t = transform(X)
        predictions = self.model.predict(X_t)
        return predictions
        pass

