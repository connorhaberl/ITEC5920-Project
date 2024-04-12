from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
from configs.wavelet_configs import *
from configs.your_configs import *
import os
from time import time

def main():

    #train and evaluate Superclass classifiers
    datafolder = '../data/'
    outputfolder = '../output/'

    models = [
        conf_random_forest,
        conf_fastai_xresnet1d101 
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [       
        ('exp1.1.1', 'superdiagnostic'),
       ]
    
    times = []
    for name, task in experiments:
        start_time = time()
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        print("finished preparing")
        e.perform()
        print("finished performing")
        e.evaluate()
        task_time = time() - start_time
        times.append(task_time)
    
    #save the time taken for training
    with open(outputfolder+"superclass_classifier_training times.txt", 'w') as output:
        for this_time in times:
            output.write(str(this_time) + '\n')
    
    #To generate mlb files for exp 1.1, 2 and 3
    #generate 
    models = [
        conf_fastai_xresnet1d101,
        ]


    experiments = [
        ('exp1.1', 'subdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
       ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        print("finished preparing")
        
    # To train class-specific subclass classifiers

    models = [conf_fastai_xresnet1d101]
    experiments = [
        ('exp1.1', 'subdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
       ]

    for subclass in ['STTC', 'NORM', 'CD', 'MI', 'HYP']:
        print('\#\#\#'+subclass+'\#\#\#')
        datafolder = '../data/Superclass_sorted_records/'+subclass+'/'
        outputfolder = '../output/'+subclass+'/'

        for name, task in experiments:
            e = SCP_Experiment(name, task, datafolder, outputfolder, models)
            e.prepare()
            print("finished preparing")
            e.perform()
            print("finished performing")
            e.evaluate()
    

   
 
    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        print("finished preparing")
        e.perform()
        print("finished performing")
        e.evaluate()
    
   

    #train and evaluate cascade classifiers

    datafolder = '../data/'
    outputfolder = '../output/cascade/'
    models = [conf_cascade_resnet_classifier,conf_cascade_classifier]
    experiments = [
        ('exp1.1', 'subdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
       ]
 
    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        print("finished preparing")
        e.perform()
        print("finished performing")
        e.evaluate()
    
    utils.generate_ptbxl_summary_table(folder = outputfolder)

if __name__ == "__main__":
    main()
