"""This scripts run testing on test dataset and returns metrics values of the models.
"""

import pandas as pd
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from utils.configuration import CONFIG
from utils.helper import *
from utils.preprocessing import *

def main():

    parser= argparse.ArgumentParser(description= "Testing Resume Classification")
    parser.add_argument("--test_data_path", type= str, default= CONFIG.test_data_path, help= "Path of the test csv")
    parser.add_argument("--model_name", type= str, default= CONFIG.model_name_or_path, \
                        help="Provide a valid huggingface language model for tokne classification")
    parser.add_argument("--model_checkpoint", type= str, default= CONFIG.model_checkpoint, help= "Path to the saved model file")
    parser.add_argument("--test_batch_size", type= int, default= CONFIG.test_batch_size, help= "Test batch size")
    parser.add_argument("--token_max_length", type= int, default= CONFIG.token_max_length, help= "Maximum sequence length.")
    
    args = parser.parse_args()
    CONFIG.test_data_path= args.test_data_path
    CONFIG.model_name_or_path= args.model_name
    CONFIG.model_checkpoint= args.model_checkpoint
    CONFIG.test_batch_size= args.test_batch_size
    CONFIG.token_max_length= args.token_max_length

    test_df= pd.read_csv(CONFIG.test_data_path)
    tokenizer= get_tokenizer(model_name= CONFIG.model_name_or_path)
    collate_fn= Collate(tokenizer= tokenizer)
    test_dataset= CustomDataset(df= test_df, tokenizer= tokenizer, cfg= CONFIG)
    test_loader= DataLoader(dataset= test_dataset,
                            batch_size= CONFIG.test_batch_size,
                            collate_fn= collate_fn,
                            num_workers= CONFIG.num_workers,
                            shuffle= False,
                            pin_memory= True,
                            drop_last= False
                            )
    
    model= ResumeClassifier(model_name= CONFIG.model_name_or_path)
    model.load_state_dict(torch.load(CONFIG.model_checkpoint, map_location= CONFIG.device))
    model.to(CONFIG.device)

    print(f"Running test for model {CONFIG.model_name_or_path}")
    f1_score= testing_loop(model= model, dataloader= test_loader, device= CONFIG.device)
    print(f"Model test f1_score: {f1_score}")
    print("Finished Testing.")

if __name__ == "__main__":
    main()

