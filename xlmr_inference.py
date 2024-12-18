import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from torch import nn
import logging
from common.dataset import *
from common.utils import *
from common.load_transformers import LoadTransformers
from common.preprocessor import Preprocessor
from common.torch_trainer import TorchTrainer
from pathlib import Path


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('xlmr_inference.log')
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

"""## Test Set"""

test_set_path = "data/b2c/test_set_final.xlsx"
try:
  test_dataset_input = {
      'data_set_path': test_set_path,
      'preprocessor': Preprocessor()
  }
  init_params = DatasetInitModel(**test_dataset_input, **dataset_labels, **dataset_params
    , **transformer_params)
  test_dataset = StanceDataset(init_params)
except ValidationError as e:
  print_log(e.json())

# We must set shuffle=False, because the order is important for adding the pre_label and explanation at the end of this file.
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print_log("test_dataloader length= " + str(len(test_dataloader.dataset)))

test_batch = next(iter(test_dataloader)) # pair_sequence_features, claim_features
assert test_batch['label'].shape[0] == test_batch['pair_sequence_features']["input_ids"].shape[0]

"""# Test"""

loss_fn = nn.CrossEntropyLoss()


trainer = TorchTrainer(None, None, None, None, loss_fn 
                 , device, accumulation_steps, print_log)

model_save_directory= "models/"

trainer.test_dataloader = test_dataloader
epoch_model_name= "/epoch6_loss0.6475212574005127.pt"
loss, result_metrics, pred_labels= trainer.test_model(model_save_directory + epoch_model_name)

print_log("loss:", loss)
print_log(result_metrics)

df_test_set= pd.read_excel(test_set_path)
df_test_set["pred_stance"]= pred_labels
df_test_set["gen_explanation"]= test_dataset.explanation

# save results in a excel file
result_save_path= f"data/{transformer_params['transformer_model_path']}/"
Path(result_save_path).mkdir(parents=True, exist_ok=True)

df_test_set.to_excel(f"{result_save_path}{epoch_model_name.replace('.pt', '')}_result_on_{test_set_path.split('/')[-1]}")
