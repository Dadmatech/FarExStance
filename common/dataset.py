from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import re
from pydantic import BaseModel, validator, ValidationError
import typing as t
from typing import Optional
from parsivar import Tokenizer
from torch import nn
from common.preprocessor import Preprocessor
import os
from os.path import exists
import re


class DatasetInitModel(BaseModel):
  """The dataset model for initializing."""

  data_set_path: str
  claim_name: str
  text_name: str
  label_name: str
  preprocessor : Preprocessor
  transformer_model_path : str
  padding: bool
  truncation: bool
  max_length: int
  claim_max_length: int
  device :torch.device
  save_load_features_path: str
  similarity_features_no_for_start: Optional[int] = 15
  similarity_features_no_for_end: Optional[int] = 5
  ner_features_no: Optional[int] = 8
  ner_model_path: str
  ner_tokenizer_path: str
  ner_none_token_id: int
  sentence_similarity_model_path: str
  remove_dissimilar_sentences: Optional[bool] = False
  similar_sentence_no: Optional[int] = 20
  log_function: object
  vectorizer: object
  remove_news_agency_name: Optional[bool] = False
  news_agency_name_path: str = ""
  news_agency_coloumn_name: str = "name"
  topk_as_explanation: Optional[int] = 2

  class Config:
    arbitrary_types_allowed = True


class DatasetItemModel(BaseModel):
  pair_sequence_features: dict
  claim_features: dict
  text_features: dict
  label: np.int64
  similarity_label: np.float64
  manual_extracted_features: np.ndarray

  class Config:
    arbitrary_types_allowed = True


class StanceDataset(Dataset):
  def __init__(self ,init_params: DatasetInitModel):

    self.data_set_path = init_params.data_set_path
    self.claim_name = init_params.claim_name
    self.text_name = init_params.text_name
    self.label_name = init_params.label_name
    self.preprocessor = init_params.preprocessor

    self.transformer_model_path = init_params.transformer_model_path
    self.__vectorizer = init_params.vectorizer
    self.padding= init_params.padding
    self.truncation= init_params.truncation
    self.max_length= init_params.max_length
    self.claim_max_length = init_params.claim_max_length

    self.device = init_params.device
    self.similarity_features_no_for_start = init_params.similarity_features_no_for_start
    self.similarity_features_no_for_end = init_params.similarity_features_no_for_end
    self.ner_features_no = init_params.ner_features_no
    self.remove_dissimilar_sentences = init_params.remove_dissimilar_sentences
    self.similar_sentence_no = init_params.similar_sentence_no

    self.ner_tokenizer = AutoTokenizer.from_pretrained(init_params.ner_tokenizer_path)
    self.ner_model = AutoModelForTokenClassification.from_pretrained(init_params.ner_model_path).to(self.device)
    self.sentence_tokenizer = Tokenizer()

    self.similarity_model = SentenceTransformer(init_params.sentence_similarity_model_path).to(self.device)
    self.cosine = nn.CosineSimilarity()
    self.log = init_params.log_function
    self.ner_none_token_id = init_params.ner_none_token_id
    self.save_load_features_path = init_params.save_load_features_path
    self.remove_news_agency_name = init_params.remove_news_agency_name
    self.topk_as_explanation= init_params.topk_as_explanation

    if self.remove_news_agency_name:
      assert (len(init_params.news_agency_name_path) > 1 and len(init_params.news_agency_coloumn_name)>1), "Set file path and coloumn name of news agency names!"
      agency_names = pd.read_excel(init_params.news_agency_name_path)[init_params.news_agency_coloumn_name]
      agency_names = agency_names.apply(self.preprocessor.clean_text)
      self.news_agency_names = agency_names.tolist()
      self.__define_news_agency_name_patterns()

    self.__read_dataset()
    self.log("-----------")
    self.log('')

    self._dataset_size = len(self.claims)

    self.__prepare_labels()

    self.__clean_dataset()
    self.log("-----------")
    self.log('')

    self.__prepare_features()
    self.log("-----------")


  def __read_dataset(self):  
      self.log("Reading dataset from: " + self.data_set_path)
    
      if ".csv" in self.data_set_path:
        self.original_df = pd.read_csv(self.data_set_path, encoding = 'utf-8')
      elif ".xlsx" in self.data_set_path:
        self.original_df = pd.read_excel(self.data_set_path)
        
      self.original_df = self.original_df.dropna()
      
      self.claims = self.original_df[self.claim_name]
      self.texts = self.original_df[self.text_name]
      self.labels = self.original_df[self.label_name]

      self.log("Class distribution: \n")
      self.log(self.original_df.groupby([self.label_name])[self.label_name].count())
      
      assert (self.claims.shape == self.texts.shape == self.labels.shape), "The features size are not equal."

      self.log("The dataset shape: " + str(self.original_df.shape))
      self.log('Reading the dataset done!')


  def __define_news_agency_name_patterns(self):
    self._news_agency_name_patterns = []
    start_words= "(به|براساس|بر اساس|بنابر|بنا بر|به نقل از)"
    report= r"((\s)*گزارش(\s)*)*"
    middle_words = r"[\u0600-\u06FF\s|\u0600-\u06FF\d+\s|\u0600-\u06FF\s\d+\s]*"
    second_type_start_words= "(خبرگزاری|روزنامه|تارنما|سایت|وبسایت|وب سایت|سرویس|پیگیری|خبرنگار|در گفتگو با|خبرنگار|در گفت و گو با|در گفتوگو با|گروه|بخش|پایگاه|در پاسخ به)?"

    for name in self.news_agency_names:
      news_agency= r"((\s)*(«|'|\")?"+name+"(»|'|\")?(\s)*(-|:|،|,|؛|\\|/|\||)*)"
      news_agency2= r"((\s)*(«|'|\")?"+name+"(»|'|\")?(\s)*((\s)*\w(\s)*)(-|:|،|,|؛|\\|/|\||)*)"
      self._news_agency_name_patterns.append("(" + start_words + report + middle_words + news_agency + "(\s)*)")
      self._news_agency_name_patterns.append("(" + second_type_start_words + middle_words + news_agency + "(\s)*)")
      self._news_agency_name_patterns.append("(" + second_type_start_words + middle_words + news_agency2 + "(\s)*)")

    self._news_agency_name_patterns.append("(تهران-|نیویورک-|Entekhab.ir|Entekhab. ir|KHAMENEI.IR|namehnews.com)")


  def __clean_dataset(self):
    self.clean_claims = self.claims.apply(self.preprocessor.clean_text)
    self.clean_texts = self.texts.apply(self.preprocessor.clean_text)
    self.log('The dataset was cleaned!')


  def __prepare_features(self):
    
    self.log("Preparing features ...")

    self.log("Tokenizing by using " + self.transformer_model_path)

    lst_target_text = []
    lst_explanation = []    

    # NER and similarity features path
    path, dataset_name = os.path.split(self.data_set_path)
    dataset_name = dataset_name.replace(".","_")
    ner_feature_path = self.save_load_features_path + "/" +dataset_name+"_ner_features_" + str(self.ner_features_no) + ".npz"
    similarity_feature_path = self.save_load_features_path + "/" +dataset_name+"_similarity_features_"+ str(self.similarity_features_no_for_start) + "_" + str(self.similarity_features_no_for_end) +".npz"
    extracted_text_path = self.save_load_features_path + "/" +dataset_name+"_extracted_text_"+str(self.max_length)+"_" + str(self.similar_sentence_no) 

    if self.remove_news_agency_name:
      extracted_text_path+= "_removed_news_agency_name"

    extracted_text_path += ".xlsx"

    # NER features
    if not exists(ner_feature_path):
      # create NER and save features
      self.log("Extracting NER features ...")
      ner_features = np.zeros((self.clean_claims.shape[0], self.ner_features_no), dtype=int)

      for index, (claim, text) in enumerate(zip(self.clean_claims.tolist(), self.clean_texts.tolist())):        
        # Could we extract these features both befor and after extracting similar sentences?
        ner_features[index] = self.__extract_ner_features(claim, text)
        if index % 1000 == 0:
          self.log(("-"*20) + str(index) + ("-"*20))

      self._ner_features = ner_features
      # Saving extracted features and sentences  
      np.savez_compressed(ner_feature_path ,self._ner_features)
    else:
      self.log("Loading NER features ...")
      self._ner_features = np.load(ner_feature_path)['arr_0']


    # similarity features
    if not exists(similarity_feature_path) or (not exists(extracted_text_path) 
        and self.remove_dissimilar_sentences):
      # create and save similarity features
      self.log("Extracting similarity features ...")
      extracted_sentence_instances = 0
      similarity_features = np.zeros((self.clean_claims.shape[0]
        , (self.similarity_features_no_for_start + self.similarity_features_no_for_end)), dtype=float)    

      for index, (claim, text, label) in enumerate(zip(self.clean_claims.tolist(), self.clean_texts.tolist(), self.labels)):
        target_text = text
        
        sentence_list, similarity_score = self.__get_claim_similarity_with_text_sentences(claim, target_text)
        similarity_features[index] = self.__extract_similarity_features(sentence_list, similarity_score)
        
        if label== 3: # if unrelated
          explanation= "ادعای وارد شده در متن گزارش نشده است."
        else:
          explanation= self.__extract_similar_sentences(sentence_list, similarity_score, self.topk_as_explanation)

        if self.remove_dissimilar_sentences and not exists(extracted_text_path):
          pair_sequence_token_count = self.__vectorizer(claim, text, return_tensors="pt")['input_ids'][0].shape[0]
          # 4 is just a threshold for special tokens
          if pair_sequence_token_count > (self.max_length + 4):
            target_text = self.__extract_similar_sentences(sentence_list, similarity_score, self.similar_sentence_no)
            extracted_sentence_instances += 1        
          
        lst_target_text.append(target_text)
        lst_explanation.append(explanation)

        if index % 1000 == 0:
          self.log(("-"*20) + str(index) + ("-"*20))

      self._similarity_features = similarity_features
      self.log("Number of instances with sentence extracted: " + str(extracted_sentence_instances))

      # remove news agency names from text
      if self.remove_news_agency_name:
        self.log("Removing news agency names from text ...")
        change_texts_no = 0
        for idx,text in enumerate(lst_target_text):
          text_len = len(text)
          for pattern in self._news_agency_name_patterns:
            text = re.sub(pattern, "", text)
          
          lst_target_text[idx] = text
          if len(text) != text_len:
            change_texts_no += 1
  
          if idx % 400 == 0:
            self.log(("-"*20) + str(idx) + ("-"*20))
            
        self.log("Number of instances with removing news agency names: " + str(change_texts_no) + "/" + str(len(lst_target_text)))

      # Saving extracted features and sentences  
      np.savez_compressed(similarity_feature_path, self._similarity_features)      
      df_target_text = pd.DataFrame(lst_target_text)
      df_target_text.columns = ["text"]
      df_target_text["explanation"]= lst_explanation
      df_target_text.to_excel(extracted_text_path)
      # End of saving extracted features and sentences
    else: # load similarity features
      self.log("Loading similarity features ...")
      self._similarity_features = np.load(similarity_feature_path)['arr_0']
      
      lst_target_text = self.clean_texts.tolist()
      saved_context_df= pd.read_excel(extracted_text_path)

      lst_explanation= saved_context_df["explanation"].tolist()      
      if self.remove_dissimilar_sentences:
        lst_target_text = saved_context_df["text"].tolist()
   
    # End of NER and similarity features

    # Tokenizing
    self._pair_sequence_features = self.__vectorizer(self.clean_claims.tolist(), lst_target_text
    , return_tensors="pt", padding= self.padding, truncation=self.truncation
    , max_length= self.max_length)

    self._text_features = self.__vectorizer(lst_target_text
    , return_tensors="pt", padding= self.padding, truncation=self.truncation
    , max_length= self.max_length)    

    self._claim_features = self.__vectorizer(self.clean_claims.tolist(), return_tensors="pt"
    , padding= self.padding, truncation=self.truncation, max_length= self.claim_max_length)       
    #End of tokenizing

    self.final_context = lst_target_text
    self.explanation = lst_explanation

    self.log(self._pair_sequence_features["input_ids"].shape[0], self._claim_features["input_ids"].shape[0], self._similarity_features.shape[0], self._ner_features.shape[0], self._dataset_size)
    assert self._pair_sequence_features["input_ids"].shape[0] == self._claim_features["input_ids"].shape[0] == self._similarity_features.shape[0] == self._ner_features.shape[0] ==  self._dataset_size, "The size of features and the dataset is not equal!"

    self.log('Preparing features done!')

  
  def __get_claim_similarity_with_text_sentences(self, claim, text):
    sentence_list = self.sentence_tokenizer.tokenize_sentences(text)

    similarity_score = []

    claim_embedding = torch.from_numpy(np.reshape(self.similarity_model.encode(claim), (1, -1)))

    for index, sentence in enumerate(sentence_list):

      sentence_embedding = torch.from_numpy(np.reshape(self.similarity_model.encode(sentence), (1, -1)))      
      similarity_score.append(self.cosine(claim_embedding, sentence_embedding).item())

    assert len(sentence_list) == len(similarity_score)

    return sentence_list, similarity_score


  def __extract_similarity_features(self, sentence_list, similarity_score):    
    
    similarity_feature = np.zeros((1, (self.similarity_features_no_for_start + self.similarity_features_no_for_end)), dtype=float)
    
    features_found_for_start = len(similarity_score[:self.similarity_features_no_for_start])
    similarity_feature[0][:features_found_for_start] = similarity_score[:self.similarity_features_no_for_start]
    
    if len(sentence_list) > self.similarity_features_no_for_start:
      # for managing features for end < self.similarity_features_no_for_end.
      features_for_end = len(sentence_list) - self.similarity_features_no_for_start
      if features_for_end > self.similarity_features_no_for_end:
        features_for_end = self.similarity_features_no_for_end
      
      similarity_feature[0][-features_for_end:] = similarity_score[-features_for_end:]


    return similarity_feature


  def __extract_similar_sentences(self, sentence_list, similarity_score, result_sent_no):
    
    # deep copy
    sorted_similarity_score = similarity_score[:]
    sorted_similarity_score.sort()
        
    # get indexs of top similar sentences
    sorted_sentence_index = []
    for score in sorted_similarity_score[-result_sent_no:]:
      sorted_sentence_index.append(similarity_score.index(score))

    # sort indexs of top similar sentences and concat all sentences in order
    extracted_text = ""
    sorted_sentence_index.sort()
    for index in sorted_sentence_index:
      extracted_text += sentence_list[index]

    return extracted_text


  def __extract_ner_features(self, claim, text):

    claim_ner_inputs = self.ner_tokenizer(claim, return_tensors="pt").to(self.device)

    text_ner_inputs = self.ner_tokenizer(text, return_tensors="pt").to(self.device)

    outputs = self.ner_model(**claim_ner_inputs)[0]
    predictions = torch.argmax(outputs, axis=2)
    
    claim_tokens = self.ner_tokenizer.tokenize(self.ner_tokenizer.decode(claim_ner_inputs['input_ids'][0]))
    text_tokens = self.ner_tokenizer.tokenize(self.ner_tokenizer.decode(text_ner_inputs['input_ids'][0]))

    ner_feature = np.zeros((1, self.ner_features_no), dtype=int)    
    last_index = 0
    for token, pred in zip(claim_tokens, predictions.T):
      if pred.item() != self.ner_none_token_id:
        ner_feature[0][last_index] = 1 if token in text_tokens else -1
        last_index += 1
        if last_index >= self.ner_features_no:
          break

    return ner_feature


  def __len__(self):
    return self._dataset_size


  def __getitem__(self, idx):
    # return pair_sequence_input_ids, pair_sequence_attention_mask, 
    # claim_input_ids, claim_attention_mask, manual_extracted_features, label, label_sim
    pair_sequence_features = {'input_ids':self._pair_sequence_features["input_ids"][idx]
                              , 'attention_mask':self._pair_sequence_features["attention_mask"][idx]}

    if 'token_type_ids' in self._pair_sequence_features.keys():
      pair_sequence_features['token_type_ids'] = self._pair_sequence_features['token_type_ids'][idx]

    claim_features = {'input_ids':self._claim_features["input_ids"][idx]
                                , 'attention_mask':self._claim_features["attention_mask"][idx]}
    if 'token_type_ids' in self._claim_features.keys():
      claim_features['token_type_ids'] = self._claim_features['token_type_ids'][idx]

    text_features = {'input_ids':self._text_features["input_ids"][idx]
                                , 'attention_mask':self._text_features["attention_mask"][idx]}    

    if 'token_type_ids' in self._text_features.keys():
      claim_features['token_type_ids'] = self._text_features['token_type_ids'][idx]

    output_features = {'pair_sequence_features': pair_sequence_features
      ,'claim_features': claim_features
      , 'text_features':text_features
      ,'label': self._target_lebel[idx]
      ,'similarity_label': self._target_label_sim[idx]
      ,'manual_extracted_features': np.concatenate((self._ner_features[idx], self._similarity_features[idx]))}

    return dict(DatasetItemModel(**output_features))


  def get_num_batches(self, batch_size):
    return len(self) // batch_size


  def __prepare_labels(self):
    # y_temp = np.zeros((self.labels.shape[0], 4), dtype=int)
    y_temp = np.zeros((self.labels.shape[0]), dtype=int)
    y_sim = np.zeros((self.labels.shape[0]), dtype=float)
    
    # To handle lebel of EN dataset (FNC)
    self.label_2_id = {'agree': 1, 'disagree': 0, 'discuss': 2, 'unrelated': 3}

    for i,item in enumerate(self.labels):
      y_sim[i] = 0
      label = item

      if item in self.label_2_id: # To handle lebel of EN dataset (FNC)
        label = self.label_2_id[item]
        
      y_temp[i] = label
      
      if (label == 1): # agree
        y_sim[i] = 1
      elif (label == 0): # disagree
        y_sim[i] = -1

    self._target_lebel = y_temp
    self._target_label_sim = y_sim


class LightDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels
    self._target_lebel = labels
    

  def __getitem__(self, idx):
    item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
    item["labels"] = torch.tensor([self.labels[idx]])
    return item

  def __len__(self):
    return len(self.labels)


class EnglishStanceDataset(Dataset):
  def __init__(self, data_set_path, log, vectorizer, max_length= 512, claim_name="header"
      , text_name="body", label_name="label"):
    
    self.log = log
    self.claim_name = claim_name
    self.text_name = text_name
    self.label_name = label_name
    self.__vectorizer = vectorizer
    self.label_2_id = {'agree': 1, 'disagree': 0, 'discuss': 2, 'unrelated': 3}

    self.read_dataset(data_set_path)

    self.__prepare_feature(max_length)


  def read_dataset(self, data_set_path):
    if ".csv" in data_set_path:
      self.original_df = pd.read_csv(data_set_path, encoding = 'utf-8')
    elif ".xlsx" in self.data_set_path:
      self.original_df = pd.read_excel(data_set_path)
      
    self.original_df = self.original_df.dropna()
    
    self.claims = self.original_df[self.claim_name]
    self.texts = self.original_df[self.text_name]
    self._target_lebel = []
    
    for label in self.original_df[self.label_name]:
        self._target_lebel.append(self.label_2_id[label])

    self.log("Class distribution: \n")
    self.log(self.original_df.groupby([self.label_name])[self.label_name].count())
    
    assert (self.claims.shape[0] == self.texts.shape[0] == len(self._target_lebel)), "The features size are not equal."

    self.log("The dataset shape: " + str(self.original_df.shape))
    self.log('Reading the dataset done!')


  def __prepare_feature(self, max_length):
    self._pair_sequence_features = self.__vectorizer(self.claims.tolist(), self.texts.tolist()
      , return_tensors="pt", padding= True, truncation=True
      , max_length= max_length)


  def __getitem__(self, idx):
    item = {k: torch.tensor(v[idx]) for k, v in self._pair_sequence_features.items()}
       
    item["labels"] = self._target_lebel[idx]

    return item


  def __len__(self):
    return len(self._target_lebel)  