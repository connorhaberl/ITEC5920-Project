import os
import shutil
import csv
import random

# Dictionary to store superclass mappings
ecg_classifications = {}
scp_mappings = {}
scp_counts = {}
valid_superclasses = ["CD", "HYP", "NORM", "MI", "STTC"]

def load_scp_mapping(scp_csv_file):
    with open(scp_csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            scp_mappings[row['label']] = row['diagnostic_class']
            scp_counts[row['label']] = 0

def add_zeros(index):
    if index < 10:
        extra_zeros = '0000'
    elif index < 100:
        extra_zeros = '000'
    elif index < 1000:
        extra_zeros = '00'
    elif index < 10000:
        extra_zeros = '0'
    else:
        extra_zeros = ''

    return extra_zeros + str(index)

# Function to load superclass mappings from CSV file
def load_superclass_mappings(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            extra_zeros = ''
            row_index = int(row['ecg_id'])
            if row_index < 10:
                extra_zeros = '0000'
            elif row_index < 100:
                extra_zeros = '000'
            elif row_index < 1000:
                extra_zeros = '00'
            elif row_index < 10000:
                extra_zeros = '0'

            ecg_id = extra_zeros + str(row_index) + "_lr.dat"
            scp_codes = eval(row['scp_codes'])
            for key in scp_codes.keys():
                scp_counts[key] = scp_counts[key] + 1
            superclass = scp_mappings[list(scp_codes.keys())[0]]  # Assuming             first key in scp_codes is the superclass
            if superclass not in valid_superclasses:
                superclass = "Unknown"
            ecg_classifications[ecg_id] = superclass

# Function to get superclass based on file name
def get_superclass(file_name):
    return ecg_classifications.get(file_name, "Unknown")

# Function to move files to appropriate folders
def resort_ecg_data(source_dir, destination_dir):
    index = 0
    for subdir, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith("_lr.dat") or file.endswith("_lr.hea"):
                src_file_path = os.path.join(subdir, file)
                
                sc_file = file[:-3]+"dat"
                superclass = get_superclass(sc_file)
                #if superclass != "Unknown":
                #superclass_folder = os.path.join(destination_dir, superclass)
                superclass_folder = os.path.join(destination_dir, superclass, "records100")
                dest_folder = os.path.join(superclass_folder, subdir.split("/")[-1])  # Get the last part of the subdir

                if index < 5:
                    a = subdir.split("/")[-1]
                    
                    print(f'sub_dir split 1: {a}')
                    
                    #print(f'sc_file: {sc_file}')
                    #print(f'superclass: {superclass}')
                    #print(f'sc_folder: {superclass_folder}')
                    #print(f'dest_folder: {dest_folder}')
                    index +=1

                os.makedirs(dest_folder, exist_ok=True)
                shutil.copy(src_file_path, dest_folder)
                #else:

                #    print(f"Unknown superclass for file: {file}")

# Function to write a new ptbxl_database.csv file within each superclass folder
def write_superclass_database(superclass_folder, superclass, csv_file_path):
    superclass_csv_file = os.path.join(superclass_folder, f"{superclass}_ptbxl_database.csv")
    with open(csv_file_path, mode='r') as file:
        with open(superclass_csv_file, mode='w', newline='') as write_file:
            reader = csv.reader(file)
            writer = csv.writer(write_file)
            header = next(reader)
            writer.writerow(header)
            index = 0
            for row in reader:
                ecg_id = add_zeros(int(row[0]))

                if ecg_classifications[ecg_id+"_lr.dat"] == superclass:
                    writer.writerow(row)

# Define base directory
base_folder = os.path.join("..","data")

source_directory = os.path.join(base_folder,"records100")  # eventually remove test_folder
destination_directory =  os.path.join(base_folder,"Superclass_sorted_records")
csv_file_path = os.path.join(base_folder,"ptbxl_database.csv")
scp_file_path = os.path.join(base_folder,"scp_statements.csv")


# Generate subclass to superclass mappings
load_scp_mapping(scp_file_path)
# Read superclasses from CSV
load_superclass_mappings(csv_file_path)
# Resort the data based on superclass
resort_ecg_data(source_directory, destination_directory)

# Iterate over each superclass folder to create ptbxl_database.csv files
for superclass_folder in os.listdir(destination_directory):
    if os.path.isdir(os.path.join(destination_directory, superclass_folder)):
        write_superclass_database(os.path.join(destination_directory, superclass_folder), superclass_folder, csv_file_path)

print("Data sorted and superclass database files created successfully.")
