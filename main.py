import torch
from torch.utils.data import DataLoader
from torchmetrics import RetrievalHitRate, RetrievalNormalizedDCG,Recall, RetrievalRecall,MeanSquaredError
from sklearn.metrics import mean_squared_error
import os,sys
import numpy as np
from bertrec_regression import BERT4REC
from transformers import Trainer,TrainingArguments
from data_utils import *
import argparse
import datetime


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignobre
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=64)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--decay_step", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=0.1)
    
    parser.add_argument('--data_dir', type=str, default="../data")
    parser.add_argument('--train_split_size', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=256)
    
    parser.add_argument('--epoch_size', type=int, default=7)

    
    

    return parser.parse_args()


def compute_metrics(p):
    predictions,labels=p
    rmse = mean_squared_error(labels, predictions, squared=False)
    
    return {"rmse": rmse}
    
def create_dataset(args,return_loader=False):
    
    ratings_df_path=args.data_dir+"/Amazon_ratings.csv"
    reviews_df_path=args.data_dir+"Amazon_reviews.csv"
        
    _,user_history_df,item2idx,user2idx,min_len =prepare_df(ratings_df_path,reviews_df_path,step_size=1,sequence_length=args.seq_len)
    train_data,valid_data=split_df(user_history_df)
    
    #train_df=pd.read_csv(args.train_data_dir)
    #valid_df=pd.read_csv(args.valid_data_dir)
    
    train_dataset=AmazoneDataset(train_data)
    valid_dataset=AmazoneDataset(valid_data)
    
    if return_loader:
    
        train_dataloader=DataLoader(train_dataset,pin_memory=True,shuffle=True,batch_size=args.batch_size)
        valid_datalaoder=DataLoader(valid_dataset,pin_memory=True,shuffle=False,batch_size=args.batch_size)
    
        return train_dataset,valid_dataset,train_dataloader,valid_dataloader,item2idx,user2idx
    
    return train_dataset,valid_dataset,item2idx,user2idx,min_len
    
def train(args):
    
    train_dataset,valid_dataset,item2idx,user2idx,min_seq_len=create_dataset(args,return_loader=False)
    item_size=len(item2idx)
    train_size=train_dataset.__len__()
    num_gpus=torch.cuda.device_count()
    
    # total eval & logging 
    
    total_num_saves=args.epoch_size * 3 # save three times per one epoch 
    total_training_steps=train_size*args.epoch_size
    eval_steps=25  #int(total_training_steps/args.batch_size/num_gpus/total_num_saves)
    
    
    # padding is not needed
    if min_seq_len>=args.seq_len:
        args.max_len=args.seq_len
        
    save_path="BERT4REC" +"_"+"seq_len_"+str(args.seq_len)+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        
    
    
    model = BERT4REC(
        vocab_size=item_size+1, # item size + pad token
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        head_num=args.head_num,
        dropout_rate=args.dropout_rate,
        initializer_range=args.initializer_range,
        user_size=len(user2idx)
        )

    training_args = TrainingArguments(
        save_total_limit=2,
        output_dir=f"./checkpoints/{save_path}/",
        evaluation_strategy="steps",
        # eval_steps=eval_steps,
        logging_steps=eval_steps,
        save_steps=eval_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch_size,
        seed=42,
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,
        overwrite_output_dir=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
if __name__=="__main__":
    args=parse_args()
    seed_everything(3)
    train(args)
