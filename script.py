
import os
import re
import shutil
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, DataCollatorWithPadding

from utils.configuration import CONFIG

from utils.helper import *
from utils.preprocessing import *

###################################
set_seed(seed= 42)

device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')
CONFIG.device= device

parser= argparse.ArgumentParser(description= "Resume Classification")
parser.add_argument("--resume_dir", type= str, default= CONFIG.resume_path, help= "Path of the test csv")
parser.add_argument("--output_dir", type= str, default= "results", help= "Path of the output files")
args = parser.parse_args()

data_root_path= args.resume_dir
output_path= args.output_dir

## Loding data and apply some preprocessin
print("Loading Data")
test_df= load_file(data_root_path)
test_df['Resume_str']= test_df['Resume_str'].apply(remove_unwanted_chars)

# Loading models and tokenizers
tokenizer= get_tokenizer(CONFIG.model_name_or_path)
model= ResumeClassifier(CONFIG.model_name_or_path)
model.load_state_dict(torch.load(CONFIG.model_checkpoint, map_location=  device))
model.to(device)

curr_dir= os.getcwd()
for folder in CONFIG.categories:
    path= os.path.join(curr_dir, output_path ,folder)
    if os.path.exists(path) and os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

filenames= test_df.ID.values
resumes= test_df.Resume_str.values
predictions= []

print("Running Predictions")
for filename, resume in zip(filenames, resumes):
    inp= tokenizer(resume, return_token_type_ids= False, truncation= True, max_length= CONFIG.token_max_length, return_tensors= "pt")
    pred= model(**inp).argmax(axis=1).to('cpu').numpy()[0]
    pred_cat= str(CONFIG.id2label[str(pred)])
    predictions.append(pred_cat)
    ### copying the file to category folder
    shutil.copy2(os.path.join(data_root_path, filename), os.path.join(curr_dir, output_path, pred_cat, filename))    

test_df["Predictions"]= predictions

test_df.to_csv("Predictions.csv", index= False)
