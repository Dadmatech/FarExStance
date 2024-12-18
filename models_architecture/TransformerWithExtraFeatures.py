import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from torch import nn
import numpy as np
from sklearn import metrics
import re
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss, MSELoss
import time, sys


"""## Model"""

# Define model
class TransformerWithExtraFeatures(nn.Module):
  def __init__(self, load_transformer_model, model_path, dens_input_size= [128,64]
    , extra_features_no= 32, num_classes=4, dens_dropout=[0.5]):

    super(TransformerWithExtraFeatures, self).__init__()

    assert len(dens_input_size) >= 2, "You should consider at least two dense layers!"
    assert len(dens_dropout)  + 1 == len(dens_input_size), "Length of dens_dropout plus one and length of dens_input_size should be equal."
    
    self.transformer_model = load_transformer_model(model_path, num_classes)

    print("Model of ", model_path ," was loaded successfully.")
   
    # Fully-connected and Dropout layers

    self.dens_list = nn.ModuleList([nn.Linear(dens_input_size[i],dens_input_size[i + 1])
        for i in range(len(dens_input_size) -1) # use the last one for classifier layer
    ])

    self.batch_norm_list = nn.ModuleList([nn.BatchNorm1d(dens_input_size[i + 1])
        for i in range(len(dens_input_size) -1)
    ])

    self.dropout_list = nn.ModuleList([nn.Dropout(p=dens_dropout[i])
        for i in range(len(dens_dropout))
    ])

    self.cosine = nn.CosineSimilarity()

    self.num_classes = num_classes
    # Plus 1 for cosine similarity feature
    self.fc = nn.Linear(dens_input_size[-1] + 1, num_classes)

  def forward(self, transformer_input_tensor, claim_tensor, text_tensor, manual_extracted_features
              , labels = None, sim_labels = None, loss_fn = None):
    
    pair_sequence_outputs = self.transformer_model(**transformer_input_tensor)

    # get vector of the [CLS] token
    pair_sequence_cls = pair_sequence_outputs["last_hidden_state"][:,0,:]

    concat_features = torch.cat([pair_sequence_cls, manual_extracted_features.float()], dim=1)
    
    dens_output = F.relu(self.dropout_list[0](self.batch_norm_list[0](self.dens_list[0](concat_features))))

    for dens, batch_norm, dropout in zip(self.dens_list[1:], self.batch_norm_list[1:], self.dropout_list[1:]):
      dens_output = F.relu(dropout(batch_norm(dens(dens_output))))
        
    claim_outputs = self.transformer_model(**claim_tensor)
    claim_cls = claim_outputs["last_hidden_state"][:,0,:]
    cos_sim = self.cosine(pair_sequence_cls, claim_cls).unsqueeze(1)
            
    combined_features = torch.cat([dens_output, cos_sim], dim=1)
    logits = self.fc(combined_features)

    loss = 10000
    prediction = torch.argmax(logits, dim=1).flatten()

    if labels is not None:
      loss = loss_fn(logits, labels)
    
    attention = pair_sequence_outputs[-1]  

    return loss, prediction, attention
