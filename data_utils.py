import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset
import random
import os


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_sequences(values, window_size, step_size):
    
    if len(values)<=window_size:
        return [values]
    
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences[:-1]


def prepare_df(ratings_df_path,reviews_df_path,step_size=1,sequence_length=None):
    
    ratings_df=pd.read_csv(ratings_df_path)
    reviews_df=pd.read_csv(reviews_df_path)
    
    reviews_df=reviews_df.drop(["index","votes","verified","review_time","user_name"],axis=1)
    merged_df=pd.merge(ratings_df,reviews_df,on=["user_id","item_id"],how="left")
    item2idx={item_id:idx+1 for idx,item_id in enumerate(merged_df["item_id"].unique())} # 0 is for padding idx
    user2idx={user_id:idx for idx,user_id in enumerate(merged_df["user_id"].unique())}
    merged_df["item_id"]=merged_df["item_id"].map(item2idx)
    merged_df["user_id"]=merged_df["user_id"].map(user2idx)
    
    item_mean_review_length=merged_df[merged_df["text"].str.len()>0].groupby("item_id")["text"].apply(lambda x: round(x.str.split().str.len().mean(),2))
    user_mean_review_length=merged_df[merged_df["text"].str.len()>0].groupby("user_id")["text"].apply(lambda x: round(x.str.split().str.len().mean(),2))
    
    merged_df["item_avg_review_length"]=merged_df["item_id"].map(item_mean_review_length)
    user_history_df=merged_df.groupby("user_id").agg(list).reset_index().drop(["timestamp","text"],axis=1)
    user_history_df["user_avg_review_length"]=user_history_df["user_id"].map(user_mean_review_length)
    
    min_len=user_history_df["item_id"].apply(len).min()
    
    if sequence_length is None:
        sequence_length=min_len
    
    user_history_df.item_id=user_history_df.item_id.apply(
        lambda ids: create_sequences(ids, sequence_length, step_size)
    )

    user_history_df.rating=user_history_df.rating.apply(
        lambda ids: create_sequences(ids, sequence_length, step_size)
    )

    user_history_df.item_avg_review_length=user_history_df.item_avg_review_length.apply(
        lambda ids: create_sequences(ids, sequence_length, step_size)

    )
    
    user_history_df=user_history_df.explode(["item_id","rating","item_avg_review_length"])
    user_history_df=user_history_df.reset_index(drop=True)
    
    return merged_df,user_history_df,item2idx,user2idx,min_len


def split_df(df,train_split_size=0.7,to_csv=False): #split user_history_df into train and valid set

    random_selection = np.random.rand(len(df)) <= train_split_size
    train_data = df[random_selection]
    valid_data = df[~random_selection]

    if to_csv:
        train_data.to_csv("amazone_data/train_data.csv", index=False)
        valid_data.to_csv("amazone_data/valid_data.csv", index=False)

    return train_data,valid_data
    

class AmazoneDataset(Dataset):
    """Amazone dataset."""

    def __init__(
        self, 
        df,
        use_review=False
    ):
        """
        Args:
            df : pandas dataframe
        """
        self.df=df
        self.use_review=use_review

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        user_id = data.user_id
        item_ids= data.item_id
        
        item_ratings = data.rating
        target_item_id = item_ids[-1:][0]
        target_item_rating = item_ratings[-1:][0]
        
        item_ratings = item_ratings[:-1]
        
        if self.use_review:
            pass
        
        input_dict={"user_id":torch.tensor(user_id),
                    "item_ids":torch.tensor(item_ids),
                    "item_ratings":torch.tensor(item_ratings),
                    "target_item_id":torch.tensor(target_item_id),
                    "labels":torch.tensor(target_item_rating),
                    }
        
        return input_dict
