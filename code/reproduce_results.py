from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
from configs.wavelet_configs import *
from configs.your_configs import *


def main():
    #physionet.org/files/ptb-xl/1.0.3/
    #class specific
    #datafolder = '../data/Superclass_sorted_records/STTC/'
    datafolder = '../data/'
    
    #datafolder_icbeb = '../data/ICBEB/'
    #class specific
    #outputfolder = '../output/latest2/STTC/'
    #outputfolder = '../output/'
    outputfolder = '../output/stateoftheart/'

    #outputfolder = '../output/latest/'
    #outputfolder = '../outputs/random_forest4/'

    models = [
        conf_fastai_xresnet1d101,
        # conf_fastai_resnet1d_wang,
        # conf_random_forest  
        #conf_cascade_classifier,
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
        ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
       ]
       
    # for subclass in ['STTC', 'NORM', 'CD', 'MI', 'HYP']:
    #     print('\#\#\#'+subclass+'\#\#\#')
    #     datafolder = '../data/Superclass_sorted_records/'+subclass+'/'
    #     outputfolder = '../output/latest2/'+subclass+'/'

    #     for name, task in experiments:
    #         e = SCP_Experiment(name, task, datafolder, outputfolder, models)
    #         e.prepare()
    #         print("finished preparing")
    #         e.perform()
    #         print("finished performing")
    #         e.evaluate()
#        utils.generate_ptbxl_summary_table(folder = outputfolder)


 
    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        print("finished preparing")
        e.perform()
        print("finished performing")
        e.evaluate()
    # generate greate summary table
    utils.generate_ptbxl_summary_table(folder = outputfolder)

    
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
