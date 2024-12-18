import torch
import torch.nn.functional as F
from torch import nn


"""## Model"""

# Define model
class DatasetBiasCheckerModel(nn.Module):
  def __init__(self, load_transformer_model, model_path, dens_input_size= [128,64]
    , num_classes=4, dens_dropout=[0.5], sequence_section= 0):
    """
     sequence_section: 0 means train model with only claims and 1 means train model with only text
    """

    super(DatasetBiasCheckerModel, self).__init__()

    assert len(dens_input_size) >= 2, "You should consider at least two dense layers!"
    assert len(dens_dropout)  + 1 == len(dens_input_size), "Length of dens_dropout plus one and length of dens_input_size should be equal."
    
    self.transformer_model = load_transformer_model(model_path, num_classes)
    self.sequence_section = sequence_section

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
    self.fc = nn.Linear(dens_input_size[-1], num_classes)

  def forward(self, transformer_input_tensor, claim_tensor, text_tensor, manual_extracted_features
              , labels = None, sim_labels = None, loss_fn = None):
    
    if self.sequence_section == 0:
      transformer_outputs = self.transformer_model(**claim_tensor)
    else:
      transformer_outputs = self.transformer_model(**text_tensor)

    # get vector of the [CLS] token
    output_cls = transformer_outputs["last_hidden_state"][:,0,:]
    
    dens_output = F.relu(self.dropout_list[0](self.batch_norm_list[0](self.dens_list[0](output_cls))))

    for dens, batch_norm, dropout in zip(self.dens_list[1:], self.batch_norm_list[1:], self.dropout_list[1:]):
      dens_output = F.relu(dropout(batch_norm(dens(dens_output))))
        
    logits = self.fc(dens_output)

    loss = 10000
    prediction = torch.argmax(logits, dim=1).flatten()

    if labels is not None:
      loss = loss_fn(logits, labels)
    
    return loss, prediction, None
