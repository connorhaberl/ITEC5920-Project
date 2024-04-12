from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
from configs.wavelet_configs import *
from configs.your_configs import *


def main():
    #select input and output folders
    datafolder = '../data/'
    outputfolder = '../output/'

    
#select models to be used
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
    #select experiments to be run
    experiments = [
        ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
       ]
    


    #run selected experiments
    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        print("finished preparing")
        e.perform()
        print("finished performing")
        e.evaluate()
    # generate greate summary table
    utils.generate_ptbxl_summary_table(folder = outputfolder)

if __name__ == "__main__":
    main()
