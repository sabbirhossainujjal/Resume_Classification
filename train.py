import pandas as pd
import numpy as np
import argparse
import pickle

from utils.configuration import CONFIG
from utils.helper import *
from utils.preprocessing import *

def main():
    ##Run full training scripts from data loading to training steps.

    # parse the args
    parser= argparse.ArgumentParser(description= "Training Resume Classification")
    parser.add_argument("--model_name", type= str, default= CONFIG.model_name_or_path, \
                        help="Provide a valid huggingface language model for tokne classification",)
    parser.add_argument("--output_dir", type= str, default= CONFIG.output_dir, help= "Path to the output model files")
    parser.add_argument("--n_folds", type= int, default= CONFIG.n_folds, help= "Number of fold to train")
    parser.add_argument("--num_epochs", type= int, default= CONFIG.num_epochs, help= "Number of epochs to run")
    parser.add_argument("--train_batch_size", type= int, default= CONFIG.train_batch_size, help= "Training batch size")
    parser.add_argument("--valid_batch_size", type= int, default= CONFIG.valid_batch_size, help= "Validation batch size")
    parser.add_argument("--learning_rate", type= float, default= CONFIG.learning_rate, help= "Initial Learning Rate")
    parser.add_argument("--scheduler", type= str, default= CONFIG.scheduler, help="Learning rate scheduler.",
                        choices=["CosineAnnealingWarmRestarts", "CosineAnnealingLR", "linear"])
    parser.add_argument("--max_length", type= int, default= CONFIG.token_max_length, help= "Maximum sequence length.")
    
    args = parser.parse_args()
    CONFIG.model_name_or_path= args.model_name
    CONFIG.output_dir= args.output_dir
    CONFIG.n_folds= args.n_folds
    CONFIG.num_epochs= args.num_epochs
    CONFIG.train_batch_size= args.train_batch_size
    CONFIG.valid_batch_size= args.valid_batch_size
    CONFIG.learning_rate= args.learning_rate
    CONFIG.scheduler= args.scheduler
    CONFIG.token_max_length= args.max_length

    #loading datasets


    ## Cross validation kfold data of the dataset
    data_df= do_cv_split(data_df= data_df)
    train_df, test_df= do_train_test_split(data_df= data_df)

    # loading tokenizer and collate function
    tokenizer= get_tokenizer(model_name= CONFIG.model_name)
    collate_fn= Collate(tokenizer= tokenizer)

    scores = []
    accuracy= []
    for fold in range(CONFIG.n_folds):
        print(f"====== Started Training Fold-{fold} ======")

        ## loading necessary data loader and model from trianing_utils.py
        train_loader, valid_loader = prepare_loader(train_df, fold= fold, cfg= CONFIG)
        model= ResumeClassifier(model_name= CONFIG.model_name_or_path)
        model.to(device= CONFIG.device)

        optimizer= get_optimizer(model.parameters(), cfg= CONFIG)
        scheduler= fetch_scheduler(optimizer= optimizer)

        # run training
        model, history, fold_best_score, fold_accuracy= training_loop(model, train_loader, valid_loader,
                                                                        optimizer, 
                                                                        scheduler,
                                                                        num_epochs= CONFIG.num_epochs, 
                                                                        fold= fold,
                                                                        patience= 3,
                                                                        cfg= CONFIG
                                                                        )
        scores.append(fold_best_score)
        accuracy.append(fold_accuracy)

        with open(f'history_fold_{fold}.pickle', 'wb') as f:
            pickle.dump(history, f)
        
        print("="*10 + f"Fold {fold} best score" + '=' * 10)
        print(f"F1 Score: {fold_best_score :.4f}")
        print(f"Accuracy: {fold_accuracy: .4f}")
        print("="* 30)

        del model, history, train_loader, valid_loader
        gc.collect()
    

    print("="*10 + "Ovaerall Performance" + '=' * 10)
    print(f"F1 Score: {np.mean(scores):.4f}")
    print(f"Accuracy: {np.mean(accuracy): .4f}")
    print("="* 30)
    
    return np.mean(scores)


if __name__ == "__main__":
    main()
