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

    #Superclass classifiers
    datafolder = '../data/'
    outputfolder = '../output/'

    models = [
        # conf_fastai_xresnet1d101,
        # conf_fastai_resnet1d_wang,
        conf_random_forest  
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        # ('exp0', 'all'),
        # ('exp1', 'diagnostic'),
        #('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        #('exp2', 'form'),
        #('exp3', 'rhythm')
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
    
    models = [
        conf_fastai_xresnet1d101,
        # conf_fastai_resnet1d_wang,
        #conf_random_forest  
        # conf_cascade_classifier,
        #conf_cascade_resnet_classifier,      
        #  conf_fastai_lstm,
        # conf_fastai_lstm_bidir,
        # conf_fastai_fcn_wang,
        # conf_fastai_inception1d,
        # conf_wavelet_standard_nn,
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        # ('exp0', 'all'),
        # ('exp1', 'diagnostic'),
        #('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        #('exp2', 'form'),
        #('exp3', 'rhythm')
       ]
    
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

    print(times)
    with open(outputfolder+"superclass_classifier_training times.txt", 'w') as output:
        for this_time in times:
            output.write(str(this_time) + '\n')
    
    #To generate mlb files for exp 1.1, 2 and 3

    models = [
        conf_fastai_xresnet1d101,
        # conf_fastai_resnet1d_wang,
        #conf_random_forest  
        # conf_cascade_classifier,
        #conf_cascade_resnet_classifier,      
        #  conf_fastai_lstm,
        # conf_fastai_lstm_bidir,
        # conf_fastai_fcn_wang,
        # conf_fastai_inception1d,
        # conf_wavelet_standard_nn,
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        # ('exp0', 'all'),
        # ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        #('exp1.1.1', 'superdiagnostic'),
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
   
    # To train class-specific subclass classifiers

    models = [conf_fastai_xresnet1d101]
    experiments = [
        # ('exp0', 'all'),
        # ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        # ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
       ]

    for subclass in ['STTC', 'NORM', 'CD', 'MI', 'HYP']:
        print('\#\#\#'+subclass+'\#\#\#')
        datafolder = '../data/Superclass_sorted_records/'+subclass+'/'
        outputfolder = '../output/latest2/'+subclass+'/'

        for name, task in experiments:
            e = SCP_Experiment(name, task, datafolder, outputfolder, models)
            e.prepare()
            print("finished preparing")
            e.perform()
            print("finished performing")
            e.evaluate()
       #utils.generate_ptbxl_summary_table(folder = outputfolder)
    # To test cascade classifiers

    datafolder = '../data/'
    outputfolder = '../output/latest2/rnn/'
    os.makedirs(outputfolder, exist_ok=True)
    
    #conf_cascade_classifier, 
    models = [your_rnn_conf]
    experiments = [
        # ('exp0', 'all'),
        # ('exp1', 'diagnostic'),
        # ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        # ('exp2', 'form'),
        # ('exp3', 'rhythm')
       ]
 
    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        print("finished preparing")
        e.perform()
        print("finished performing")
        e.evaluate()
    
    # generate greate summary table
    utils.generate_ptbxl_summary_table(folder = outputfolder)


    # datafolder = '../data/'
    # outputfolder = '../output/latest2/cascade/'
    # #conf_cascade_classifier, 
    # models = [conf_cascade_resnet_classifier]
    # experiments = [
    #     # ('exp0', 'all'),
    #     # ('exp1', 'diagnostic'),
    #     ('exp1.1', 'subdiagnostic'),
    #     # ('exp1.1.1', 'superdiagnostic'),
    #     ('exp2', 'form'),
    #     ('exp3', 'rhythm')
    #    ]
 
    # for name, task in experiments:
    #     e = SCP_Experiment(name, task, datafolder, outputfolder, models)
    #     e.prepare()
    #     print("finished preparing")
    #     e.perform()
    #     print("finished performing")
    #     e.evaluate()
    
    # # generate greate summary table
    # utils.generate_ptbxl_summary_table(folder = outputfolder)

   


    # ##########################################
    # # EXPERIMENT BASED ICBEB DATA
    # ##########################################

    # # e = SCP_Experiment('exp_ICBEB', 'all', datafolder_icbeb, outputfolder, models)
    # # e.prepare()
    # # e.perform()
    # # e.evaluate()

    # generate greate summary table
    # utils.ICBEBE_table()

if __name__ == "__main__":
    main()
