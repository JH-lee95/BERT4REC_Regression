import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from transformers import AutoModel
from copy import deepcopy
import gc



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_len: int, dropout_rate: float = 0.1):
        """
        :param vocab_size: total_vocab_size
        :param embed_size: embedding size of token embedding
        :param max_len : max_len of sequence
        :param dropout_rate: dropout rate
        """
        super(BERTEmbeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_len, embed_size)
        self.segment_embeddings = nn.Embedding(3, embed_size, padding_idx=0)
        # layer_norm + dropout
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, seq: torch.Tensor, segment_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = seq.size(0), seq.size(1)  # seq : (batch, seq_len)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # position_ids : (batch_size, seq_length)
        # token , position embeddings
        token_embeddings = self.token_embeddings(seq)
        position_embeddings = self.position_embeddings(position_ids)
        # bert_embedddings
        embeddings = token_embeddings + position_embeddings
        # segment embeddings
        if segment_label is not None:
            segment_embeddings = self.segment_embeddings(segment_label)
            embeddings += segment_embeddings
        # layer-norm + drop out
        embeddings = self.dropout(self.layer_norm(embeddings))

        return embeddings


class BERT4REC(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(
        self,
        vocab_size: int = 20695+2,
        max_len: int = 8,
        hidden_dim: int = 256,
        head_num: int = 12,
        dropout_rate: float = 0.1,
        initializer_range: float = 0.02,
        use_review_info=False,
        user_size=None,
        user_emb_dim=64,
        
    ):
        """
        :param vocab_size: vocab_size of total words
        :max_len : max len of seq
        :param hidden_dim: BERT model hidden size
        :param head_num : number of attention heads
        :param dropout_rate : dropout rate
        :param dropout_rate_attn : attention layer의 dropout rate
        :param initializer_range : weight initializer_range
        """
        super(BERT4REC, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.ff_dim = hidden_dim * 4
        self.user_size=user_size
        self.user_emb_dim=user_emb_dim
        # embedding
        
        
        self.item_embedding = BERTEmbeddings(vocab_size=self.vocab_size, embed_size=self.hidden_dim, max_len=self.max_len)
        self.user_embedding=nn.Embedding(self.user_size,self.user_emb_dim)
        self.transformer_encoders = nn.TransformerEncoderLayer(d_model=self.hidden_dim, 
                                                               nhead=self.head_num,
                                                               dim_feedforward=self.ff_dim,
                                                               dropout=self.dropout_rate,
                                                               batch_first=True,)
        
        self.output_layer = nn.Sequential(
            nn.Linear(
                self.user_emb_dim+self.hidden_dim*self.max_len+self.max_len-1, #user dim, item_dim, sequence size, input rating size 
                self.hidden_dim*4,
            ),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(self.hidden_dim*4, self.hidden_dim*2),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(self.hidden_dim, 1),
        )
        
        # weight initialization
        self.initializer_range = initializer_range
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self, 
        user_id=None,
        item_ids=None,
        item_ratings=None,
        target_item_id=None,
        labels=None,
        segment_info: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        
        # mask : [batch_size, seq_len] -> [batch_size, 1, seq_len] -> [batch_size, 1, 1, seq_len]
        mask = (user_id > 0).unsqueeze(1).unsqueeze(1)
        item_embs = self.item_embedding(item_ids, segment_info)
        user_embs=self.user_embedding(user_id)
            
        #for transformer in self.transformer_encoders:
        #    transformer_output = transformer(item_embs, mask)
        
        transformer_output = self.transformer_encoders(item_embs)
            
        transformer_output=torch.flatten(transformer_output,start_dim=1)
        concat_features=torch.cat([transformer_output,user_embs,item_ratings],dim=-1)

        logits=self.output_layer(concat_features) # logits : [batch_size,1]
        
        #print("logits shape: " , logits.shape)
        
        if labels is not None:
            loss_fn=self.criterion = torch.nn.MSELoss()
            loss = loss_fn(logits.view(-1), labels)
            
            return loss,logits
            
        return logits
