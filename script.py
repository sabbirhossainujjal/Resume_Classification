
import os
import re
import numpy as np
import pandas as pd

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, DataCollatorWithPadding

from utils.configuration import CONFIG

from utils.helper import *
from utils.preprocessing import *

###################################
set_seed(seed= 42)

device= torch.device("cuda" if torch.cuda.is_available else 'cpu')
CONFIG.device= device

# model_checkpoint= "Models/debarta_v3/Resume_Classification-0.bin"
# CONFIG.model_checkpoint= model_checkpoint

data_path= "/media/sabbir/E/Research/Resume_classification_task/Resume.csv" #"/kaggle/input/resume-dataset/Resume/Resume.csv"
data_df= pd.read_csv(data_path)

data_df.drop(columns= ["Resume_html"], axis= 1, inplace= True)

## removing unwanted characters from the text
data_df["Resume_str"]= data_df["Resume_str"].apply(remove_unwanted_chars)
data_df["labels"]= data_df['Category'].apply(lambda x: int(CONFIG.label2id[x]))


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_df= do_cv_split(data_df= data_df)
train_df, test_df= do_train_test_split(data_df= data_df)


tokenizer= get_tokenizer(CONFIG.model_name_or_path)
collate_fn= Collate(tokenizer= tokenizer) #DataCollatorWithPadding(tokenizer, 
                                    # padding= True, 
                                    # max_length= CONFIG.token_max_length, 
                                    # return_tensors= 'pt')

test_dataset= CustomDataset(test_df, tokenizer, CONFIG)
test_loader= DataLoader(test_dataset, 
                        batch_size= CONFIG.test_batch_size,
                        collate_fn= collate_fn, 
                        num_workers= CONFIG.num_workers,
                        shuffle= False,
                        pin_memory= True,
                        drop_last= False,
                        )

model= ResumeClassifier(CONFIG.model_name_or_path)
# model.load_state_dict(torch.load(CONFIG.model_checkpoint, map_location=  device))
model.to(device)

id2label= CONFIG.id2label
predictions, test_f1_score, test_accuracy= testing_loop(model, test_loader, device)
predictions= [id2label[str(p)] for p in predictions]
test_df['Model_predictions']= predictions
test_df= test_df[["ID", "Resume_str", "Category", "Model_predictions" ]]
test_df.to_csv("Results.csv", index= False)
