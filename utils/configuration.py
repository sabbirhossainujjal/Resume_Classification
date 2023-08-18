

categories= ['ACCOUNTANT','ADVOCATE', 'AGRICULTURE','APPAREL','ARTS','AUTOMOBILE','AVIATION','BANKING','BPO','BUSINESS-DEVELOPMENT','CHEF', 'CONSTRUCTION', 'CONSULTANT', 'DESIGNER','DIGITAL-MEDIA','ENGINEERING','FINANCE','FITNESS', 'HEALTHCARE','HR',
    'INFORMATION-TECHNOLOGY','PUBLIC-RELATIONS','SALES','TEACHER']

label2id= {}
id2label= {}
for i, cat in enumerate(categories):
    label2id[cat] = i
    id2label[str(i)]= cat

class CONFIG:
    train= False #False #True
    test = False
    seed= 42
    n_fold= 3
    categories= categories
    id2label= id2label
    label2id= label2id
    important_punctuations= [p for p in "()-.:;?/_{|}"]
    num_labels= len(categories)
    n_folds= 3
    num_epochs= 50
    
    data_path= "data/Resume.csv"
    train_data_path= "data/train_data.csv"
    test_data_path= "data/test_data.csv"
    resume_path= "data/Resumes"
    output_dir= "Models/"
    model_name_or_path= "microsoft/deberta-v3-small"  #"bert-base-uncased" #"xlnet-base-cased" #"microsoft/deberta-v3-base" # "microsoft/deberta-v3-small" #"bert-large-uncased" #"bert-base-uncased" # #"bert-base-uncased" # #"prajjwal1/bert-small"
    model_checkpoint= "models/deverta_v3_small/Resume_Classification_fold-0.bin" # This have to be changed with proper path
    token_max_length= 512 #512 # 1024 #bert-base-uncased must be run with 512 length
    train_batch_size= 4
    valid_batch_size= 4
    test_batch_size= 4
    num_workers= 2
    
    patience= 3
    gradient_accumulation_steps= 1
    learning_rate= 2e-5 #5e-5
    weight_decay= 1e-1
    scheduler= "CosineAnnealingWarmRestarts" #"linear"
    T_max= 500
    T_0= 500
    min_lr= 1e-8
    eps = 1e-6
    betas= [0.9, 0.999]
