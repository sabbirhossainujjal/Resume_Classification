

############################## 
#imports
import time
import os
import random
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from dotmap import DotMap
import string
from collections import defaultdict

import torch
import torch.optim as optim 
from torch.optim import AdamW, lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from datasets import Dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer, DataCollatorWithPadding

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from utils.configuration import CONFIG

##################################################
CONFIG= CONFIG()

def set_seed(seed= 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False
    os.environ['PYTHONHASHSEED'] = str(seed)



def do_cv_split(data_df):
    skf = StratifiedKFold(n_splits= CONFIG.n_fold + 1, shuffle= True, random_state= CONFIG.seed)

    for fold, (_, val_) in enumerate(skf.split(X= data_df, y= data_df['labels'])):
        data_df.loc[val_, "kfold"] = int(fold)
    data_df['kfold']= data_df['kfold'].astype(int)

    return data_df

def do_train_test_split(data_df):
    test_df= data_df[data_df['kfold']== 3].reset_index(drop= True)
    train_df= data_df[data_df['kfold'] != 3].reset_index(drop= True)

    print(f"Total data in train dataset: {len(train_df)}")
    print(f"Total data in test dataset: {len(test_df)}")
    train_df.to_csv("./data/train_data.csv", index= False)
    print("Train Data is saved to './data/train_data.csv'")
    test_df.to_csv("./data/test_data.csv", index= False)
    print("Test Data is saved to '/data/test_data.csv'")

    return train_df, test_df


#########################################
# Dataset and dataloader
def get_tokenizer(model_name= CONFIG.model_name_or_path):
    return  AutoTokenizer.from_pretrained(model_name)


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, cfg= CONFIG):
        self.df= df
        self.max_len= cfg.token_max_length
        self.tokenizer= tokenizer
        # self.full_text= df['Resume_str'].values
        # self.labels= df['labels'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text= self.df.Resume_str[index] #full_text[index]
        labels= self.df.labels[index]
        inputs= self.tokenizer( text,
                                truncation= True,
                                add_special_tokens= True,
                                max_length= self.max_len,
                                padding= True, 
                            )  
        
        input_ids= inputs['input_ids']
        if len(input_ids) > self.max_len -1 :
            input_ids= input_ids[:self.max_len -1]
        
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        
        attention_mask= [1] * len(input_ids)
        labels= self.labels[index]
        
        return {
            'input_ids' : input_ids,
            'attention_mask' : attention_mask,
            'targets' : labels
        }

class Collate:
    """Data collator class for creating batch with equal length of input_ids and labels which are expected for language models
    """
    def __init__(self, tokenizer):
        self.tokenizer= tokenizer
    
    def __call__(self, batch):
        output= dict()
        output["input_ids"] = [sample['input_ids'] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output['targets'] = [sample['targets'] for sample in batch]
        
        batch_max= max([len(ids) for ids in output['input_ids']])
        
        # dynamic padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [ids + (batch_max - len(ids))*[self.tokenizer.pad_token_id] for ids in output['input_ids']]
            output['attention_mask']= [mask + (batch_max - len(mask))*[0] for mask in output['attention_mask']]
            output['targets']= [target + (batch_max - len(target))*[-100] for target in output['targets']]
        else:
            output["input_ids"] = [(batch_max - len(ids))*[self.tokenizer.pad_token_id] + ids for ids in output['input_ids']]
            output['attention_mask']= [(batch_max - len(mask))*[0] + mask for mask in output['attention_mask']]
            output['targets']= [(batch_max - len(target))*[-100] + target for target in output['targets']]
        
        # convert array to tensor
        output["input_ids"] = torch.tensor(output['input_ids'], dtype= torch.long)
        output["attention_mask"] = torch.tensor(output['attention_mask'], dtype= torch.long)
        output['targets'] = torch.tensor(output['targets'], dtype=torch.long)#
        
        return output

def prepare_loader(df, tokenizer, fold, cfg= CONFIG):
    collate_fn= Collate(tokenizer) #DataCollatorWithPadding(tokenizer, padding= True, max_length= cfg.token_max_length, return_tensors= 'pt')
    
    df_train= df[df.kfold != fold].reset_index(drop= True) # 3 fold out of 4 fold is used as training data, and 1 fold for validation.
    df_valid= df[df.kfold == fold].reset_index(drop= True)
    valid_labels = df_valid['labels'].values
    
    # converting dataFrame to dataset.
    train_dataset= CustomDataset(df_train, tokenizer, cfg)
    valid_dataset= CustomDataset(df_valid, tokenizer, cfg)
    
    train_loader= DataLoader(train_dataset, 
                            batch_size= cfg.train_batch_size, 
                            collate_fn= collate_fn, #merges a list of samples to form a mini-batch of Tensors
                            num_workers= cfg.num_workers, 
                            shuffle= True, 
                            pin_memory= True,
                            drop_last= False, )
    
    valid_loader= DataLoader(valid_dataset, 
                            batch_size= cfg.valid_batch_size,
                            collate_fn= collate_fn, 
                            num_workers= cfg.num_workers,
                            shuffle= False,
                            pin_memory= True, 
                            drop_last= False,
                            )
    
    return train_loader, valid_loader 

################################################
## Model defination

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded= attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings= torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask= input_mask_expanded.sum(1)
        sum_mask= torch.clamp(sum_mask, min= 1e-9)
        mean_embeddings= sum_embeddings/sum_mask
        return mean_embeddings
    

class ResumeClassifier(nn.Module):
    def __init__(self, model_name = CONFIG.model_name_or_path):
        super(ResumeClassifier, self).__init__()
        
        self.num_labels= CONFIG.num_labels
        self.config= AutoConfig.from_pretrained(model_name) 
        self.model= AutoModel.from_pretrained(model_name, config= self.config)
        self.model.gradient_checkpointing_enable() #for gradient checkpointing.
        
        self.drop= nn.Dropout(0.0)
        self.pooler= MeanPooling()
        self.fc= nn.Linear(self.config.hidden_size, self.num_labels)
        
        
    def forward(self, input_ids, attention_mask): 
        out= self.model(input_ids= input_ids, 
                        attention_mask= attention_mask,
                        output_hidden_states= True,)
        out= self.pooler(out.last_hidden_state, attention_mask)
        outputs= self.fc(out)
        
        return outputs
    

##########################################
# metrics and optimizer
def get_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average= 'macro')
def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_optimizer(parameters, cfg= CONFIG):
    return AdamW(params=parameters, lr= cfg.learning_rate, weight_decay= cfg.weight_decay, eps= cfg.eps, betas= cfg.betas)

def fetch_scheduler(optimizer, cfg= CONFIG):
    """
    Gets leanring rate schedular for given optimizer
    """
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler= lr_scheduler.CosineAnnealingLR(optimizer, T_max= cfg.T_max, eta_min= cfg.min_lr)
    elif cfg.scheduler == "CosineAnnealingWarmRestarts":
        scheduler= lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0= cfg.T_0, eta_min= cfg.min_lr)
    elif cfg.scheduler== "linear":
        scheduler= lr_scheduler.LinearLR(optimizer, start_factor= 0.01, end_factor= 1.0, total_iters= 100)
    elif cfg.scheduler == None:
        return None

    return scheduler


############################
# training, validation, testing functions

def train_one_epoch(model, optimizer, scheduler, dataloader, epoch, device):
    
    model.train()
    dataset_size= 0
    running_loss= 0.0
    
    steps= len(dataloader)
    bar= tqdm(enumerate(dataloader), total= len(dataloader))
    
    for step, data in bar:
        # sending data to cpu or gpu if cuda avaiable.
        ids= data["input_ids"].to(device, dtype= torch.long)
        masks= data["attention_mask"].to(device, dtype= torch.long)
        targets= data["targets"].to(device, dtype= torch.long)
        
        batch_size= ids.size(0)
        
        #computing model output
        outputs= model(ids, masks)
        
        #loss calcuation
        loss= nn.CrossEntropyLoss()(outputs, targets)
        if CONFIG.gradient_accumulation_steps > 1:
            loss= loss / CONFIG.gradient_accumulation_steps
        
        loss.backward()
        
        ## Gradient Accumulation
        if (step + 1) % CONFIG.gradient_accumulation_steps == 0 or step == steps - 1:
            
            optimizer.step() #Performs a single optimization step (parameter update)
            
            # clear out the gradients of all Variables 
            # in this optimizer (i.e. W, b)
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss= running_loss / dataset_size
        
        bar.set_postfix(Epoch= epoch, 
                        Train_Loss = epoch_loss,
                        LR= optimizer.param_groups[0]['lr'],
                        ) 
    return epoch_loss



def valid_one_epoch(model, dataloader, epoch, device):
    model.eval()
    
    dataset_size= 0
    running_loss= 0.0
    
    preds= []
    labels= []

    bar= tqdm(enumerate(dataloader), total= len(dataloader))
    
    for step, data in bar:
        ids= data["input_ids"].to(device, dtype= torch.long)
        masks= data["attention_mask"].to(device, dtype= torch.long)
        targets= data["targets"].to(device, dtype= torch.long)
        
        batch_size= ids.size(0)
        with torch.no_grad():
            outputs= model(ids, masks)
            
            loss= nn.CrossEntropyLoss()(outputs, targets)
            
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss= running_loss / dataset_size
        
        preds.append(outputs.argmax(axis=1).to('cpu').numpy())
        labels.append(targets.to('cpu').numpy())
        bar.set_postfix(Epoch= epoch, 
                        Valid_Loss = epoch_loss,
                        )
    predictions= np.concatenate(preds)
    true_labels= np.concatenate(labels)
    f1score= get_score(true_labels, predictions)
    accuracy= get_accuracy(true_labels, predictions)
    
    return epoch_loss, f1score, accuracy

################################################
def training_loop(model, train_loader, valid_loader, optimizer, scheduler, fold, num_epochs, cfg= CONFIG, patience= 3):

    if torch.cuda.is_available():
        print("Training with GPU\n")
    else:
        print("Training with CPU \n")
    
    start= time.time()
    best_score= - np.inf
    trigger_times= 0 # for early stoping
    
    history= defaultdict(list)

    for epoch in range(1, num_epochs + 1):

        train_epoch_loss= train_one_epoch(model, optimizer, scheduler, dataloader= train_loader, epoch= epoch, device= cfg.device)
        val_epoch_loss, f1score, accuracy= valid_one_epoch(model, valid_loader, epoch= epoch, device= cfg.device)
        
        history['train_loss'].append(train_epoch_loss)
        history['valid_loss'].append(val_epoch_loss)
        history['F1_score'].append(f1score)
        history['Accuracy'].append(accuracy)


        if f1score >= best_score:
            
            trigger_times= 0 #for early stop
            print(f"Validation Score Improved ({best_score :.4f} ---> {f1score :.4f})")
            print(f"Validation Accuracy: {accuracy :.4f}")
            
            best_score= f1score

            # copy and save model
            best_model_wts= copy.deepcopy(model.state_dict())
            PATH= f"Resume_Classification_fold-{fold}.bin"
            torch.save(model.state_dict(), PATH)
            print(f"Model Saved to {PATH}")
        
        else:
            trigger_times += 1
            
            if trigger_times >= patience:
                print("Early stoping \n")
                break
        
    end= time.time()
    time_elapsed= end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    
    print(f"Best F1 Score: {best_score:.4f}")
    
    # load best model weights
    model.load_state_dict(best_model_wts)


    return model, history, best_score, accuracy

def testing_loop(model, dataloader, device):
    model.eval()
    
    preds= []
    labels= []

    bar= tqdm(enumerate(dataloader), total= len(dataloader))
    
    for step, data in bar:
        ids= data["input_ids"].to(device, dtype= torch.long)
        masks= data["attention_mask"].to(device, dtype= torch.long)
        targets= data["targets"].to(device, dtype= torch.long)
        
        with torch.no_grad():
            outputs= model(ids, masks)
            score= get_score(np.concatenate(labels), np.concatenate(preds))
            preds.append(outputs.argmax(axis=1).to('cpu').numpy())
            labels.append(targets.to('cpu').numpy())
        
        bar.set_postfix(F1_score= score )

    predictions= np.concatenate(preds)
    true_labels= np.concatenate(labels)
    f1score= get_score(true_labels, predictions)
    accuracy= get_accuracy(true_labels, predictions)

    print(f"====== xxxx ======")
    print(f"Overall F1 Score: {f1score}")
    print(f"Overall  Accuracy: {accuracy}")
    print(f"====== xxxx ======")
    
    return predictions, f1score, accuracy


