import pandas as pd
import numpy as np
import ast
import wfdb
from tqdm import tqdm
from utils import utils
datafolder = '../data/ptbxl/'
sampling_frequency = 100

data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)

concatenated_data = [np.concatenate(patient_data, axis=0) for patient_data in data]

# Preprocess label data

# Select relevant data and convert to one-hot
print(len(concatenated_data))
print(len(concatenated_data[0]))