from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, default_data_collator, get_scheduler
import argparse
from datasets import load_dataset
# from transformers.models.bartpho.tokenization_bartpho_fast import BartphoTokenizerFast
from transformers import AutoModelForQuestionAnswering, default_data_collator, get_scheduler
from torch import nn
# import evaluate
import numpy as np
from torch.optim import AdamW, Adam
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from transformers.optimization import SchedulerType
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging

class Rational_Tagging(nn.Module):
    def __init__(self,  hidden_size):
        super(Rational_Tagging, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, 1)

    def forward(self, h_t):
        h_1 = self.W1(h_t)
        h_1 = F.relu(h_1)
        p = self.w2(h_1)
        p = torch.sigmoid(p)
        return p
    
class RTLoss(nn.Module):
    
    def __init__(self, hidden_size = 768, device = 'cuda'):
        super(RTLoss, self).__init__()
        self.device = device
    
    def forward(self, pt: torch.Tensor, Tagging:  torch.Tensor):
        '''
        Tagging: list paragraphs contain value token. If token of the paragraphas is rationale will labeled 1 and other will be labeled 0 
        
        RT: 
                    p^r_t = sigmoid(w_2*RELU(W_1.h_t))
            
            With:
                    p^r_t constant
                    w_2 (d x 1)
                    W_1 (d x d)
                    h_t (1 x d)
                    
            This formular is compute to each token in paraphase. I has convert into each paraphase
            
                    p^r_t = sigmoid(w_2*RELU(W_1.h))
                    
                    With:
                            p^r (1 x n) with is number of paraphase
                            w_2 (d x 1)
                            W_1 (d x d)
                            h (n x d) 
                            
        '''
        
        Tagging = torch.tensor(Tagging, dtype=torch.float32).to(device)
                
        total_loss = torch.tensor(0, dtype= torch.float32).to(device)
        
        N = pt.shape[0]
                
        for i, text in enumerate(pt):
            T = len(Tagging[i])
            Lrti = -(1/T) * (Tagging[i]@torch.log(text) + (1.0 - Tagging[i]) @ torch.log(1.0 - text) )[0]
            total_loss += Lrti
            
        return total_loss/N

class BaseLoss(nn.Module):
    
    def __init__(self):
        super(BaseLoss, self).__init__()
    
    def forward(self, start_logits: torch.Tensor, end_logits: torch.Tensor, start_positions:torch.Tensor , end_positions:torch.Tensor ):
        
        batch_size = start_logits.shape[0]
        
        start_zero = torch.zeros(start_logits.shape)
        end_zero = torch.zeros(start_logits.shape)

        for batch, y in enumerate(start_positions):
            start_zero[batch][y][0] = 1
            
        for batch, y in enumerate(end_positions):
            end_zero[batch][y][0] = 1

        start_logits = F.softmax(start_logits, dim=1)
        end_logits = F.softmax(end_logits, dim=1)
        
        proba_start = (start_logits*start_zero).sum(dim=1) 
        proba_end = (end_logits*end_zero).sum(dim=1) 

        proba_start = torch.log(proba_start )
        proba_end = torch.log(proba_end )

        total_loss = -(1/(2*batch_size)) * torch.sum(proba_start + proba_end)
        
        return total_loss
class comboLoss(nn.Module):
    def __init__(self, alpha: int = 1, beta: int = 1):
        
        super(comboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.BaseLoss = BaseLoss()
        self.RTLoss = RTLoss()
        
    def forward(self, output: dict):

        start_logits = output['start_logits']
        end_logits = output['end_logits']
        
        start_positions = output['start_positions']
        end_positions = output['end_positions']
        
        Tagging = output['Tagging']
        pt = output['pt']
        
        loss_base = self.BaseLoss(start_logits = start_logits, end_logits = end_logits, start_positions = start_positions, end_positions = end_positions)
        retation_tagg_loss  = self.RTLoss(pt = pt, Tagging = Tagging)
        
        total_loss = self.alpha*loss_base + self.beta*retation_tagg_loss
        
        return total_loss