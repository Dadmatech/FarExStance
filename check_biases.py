import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import logging
from common.dataset import *
from common.utils import *
from common.load_transformers import LoadTransformers
from common.preprocessor import Preprocessor
from common.torch_trainer import TorchTrainer
from models_architecture.DatasetBiasCheckerModel import DatasetBiasCheckerModel


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('check_biases.log')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
logger.addHandler(file_handler)
# End of logging

def print_log(*args):
    '''
    This function print and log all input arguments into a file.
    '''      
    for arg in args:
        print(arg)
        logger.info(arg)


device = None
print_log("-"*20)
if torch.cuda.is_available():
  device = torch.device('cuda')
  print_log("Device is cuda.\n")
else:
  device = torch.device('cpu')
  print_log("Device is cpu.\n")

# change to 1 for checking the biases in news articles.
BIAS_TYPE= 0
BIAS_TYPE_desc= ["Claims", "News Articles"]

print_log(f"Start checking the biases of {BIAS_TYPE_desc[BIAS_TYPE]}!")

"""# Read Dataset"""

ner_model_id= 'HooshvareLab/bert-base-parsbert-ner-uncased'
dataset_labels = {'claim_name': "claim", 'text_name': "content",'label_name': "stance"}

dataset_params = {'padding' : True, 'truncation': True, 'max_length': 512
        , 'claim_max_length': 38, 'device': device
        , 'similarity_features_no_for_start': 5
        , 'similarity_features_no_for_end': 2, 'ner_features_no': 8
        , 'remove_dissimilar_sentences': True, 'similar_sentence_no': 8
        , 'ner_model_path': 'HooshvareLab/bert-base-parsbert-ner-uncased' # for this model 0 is None
        , 'ner_tokenizer_path': 'HooshvareLab/bert-base-parsbert-ner-uncased'
        , 'ner_none_token_id': 0 
        , 'sentence_similarity_model_path':  "sentence-transformers/all-MiniLM-L12-v2" #'myrkur/sentence-transformer-parsbert-fa'
        , 'log_function': print_log
        , 'save_load_features_path': 'data/features'
        , 'remove_news_agency_name': False
        , 'news_agency_name_path':'data/news_agency_names.xlsx'
        , 'news_agency_coloumn_name':'name'}

transformer_params = {'model_loader': LoadTransformers.xlm_roberta
        , 'transformer_model_path' : "FacebookAI/xlm-roberta-large"}

vectorizer = AutoTokenizer.from_pretrained(transformer_params["transformer_model_path"])   
transformer_params["vectorizer"] = vectorizer

batch_size = 32
accumulation_steps = 1

"""## Train Set"""

train_set_path = "data/b2c/train_set_final.xlsx"

try:
  trin_dataset_input = {
      'data_set_path': train_set_path,
      'preprocessor': Preprocessor()
  }
  init_params = DatasetInitModel(**trin_dataset_input, **dataset_labels, **dataset_params
    , **transformer_params)
  train_dataset = StanceDataset(init_params)
except ValidationError as e:
  print_log(e.json())

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print_log("train_dataloader length= " + str(len(train_dataloader.dataset)))

batch = next(iter(train_dataloader)) # pair_sequence_features, claim_features
assert batch['label'].shape[0] == batch['pair_sequence_features']["input_ids"].shape[0]

"""## Validation Set"""
val_set_path = "data/b2c/dev_set_final.xlsx"

try:
  val_dataset_input = {
      'data_set_path': val_set_path,
      'preprocessor': Preprocessor()
  }
  init_params = DatasetInitModel(**val_dataset_input, **dataset_labels, **dataset_params
    , **transformer_params)
  val_dataset = StanceDataset(init_params)
except ValidationError as e:
  print_log(e.json())

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print_log("val_dataloader length= " + str(len(val_dataloader.dataset)))

val_batch = next(iter(val_dataloader))
assert val_batch['label'].shape[0] == val_batch['pair_sequence_features']["input_ids"].shape[0]


"""# Train"""
# 1024 = the shape aize of CLS

extra_features_no = (dataset_params["similarity_features_no_for_start"] 
      + dataset_params["similarity_features_no_for_end"] + dataset_params["ner_features_no"])
first_dense_input_size = (1024 + extra_features_no)

model = DatasetBiasCheckerModel(transformer_params['model_loader'] 
      ,transformer_params['transformer_model_path'], dens_input_size= [1024, 128, 32]
      , num_classes=4, dens_dropout=[0.4, 0.3], sequence_section=BIAS_TYPE)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 3e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.8e-5)
num_epochs = 30
patience = 3

trainer = TorchTrainer(model, train_dataloader, val_dataloader, optimizer, loss_fn 
                 , device, accumulation_steps, print_log)

if BIAS_TYPE==0:
  model_save_directory= f"models/claim_bias/"
else:
  model_save_directory= f"models/news_article_bias/"

loss_log = trainer.fit(num_epochs, patience, model_save_directory)