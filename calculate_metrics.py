from common.utils import *
import argparse
import pandas as pd
from pathlib import Path
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from abc import ABC, abstractmethod
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from parsivar import Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import logging
import nltk
import regex as re


nltk.download('punkt')
nltk.download('punkt_tab')

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
# End of logging

# logging
file_handler = logging.FileHandler('calculate_metrics.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# End of logging


def log(*args):
    '''
    This function print and log all input arguments into a file.
    '''      
    for arg in args:
        print(arg)
        logger.info(arg)


NLI_LABEL_ID= {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

stance_labels= {
    "disagree": 0,
    "agree": 1,
    "discuss": 2,
    "unrelated": 3,
    "other": 4
}

sentence_tokenizer= Tokenizer()
def sent_tokenize(input_text):

    return sentence_tokenizer.tokenize_sentences(input_text)


def is_farsi(text):

    pattern = r'[^a-zA-Z]+'
    regex = re.compile(pattern)
    matched = 0
    not_matched = 0
    
    for word in text.split():
        if regex.match(word):
            matched += 1
        else:
            not_matched += 1
            
    # Return True if the text matches the pattern, False otherwise
    return matched > not_matched


class NLIStructure(ABC):
    @abstractmethod
    def predict_nli(self, premise, hypothesis):
        ''' The implementation of this function should return the NLI label ID with "entailment": 0, "neutral": 1, and "contradiction": 2.

        :param premise: The premise
        :type premise: str
        :param hypothesis: The hypothesis
        :type hypothesis: str
        
        :returns: The NLI label ID ("entailment": 0, "neutral": 1, and "contradiction": 2)
        :rtype: int
        '''        
        pass


class NLEMetrics():
    '''
    The NLEMetrics object is responsible for obtaining different metrics to evaluate a generated explanation.

    :param pred_list: The list of generated explanation
    :type pred_list: list
    :param target_list: The list of ground truth explanation
    :type target_list: list
    :param claim_list: The list of claims
    :type claim_list: list
    :param claim_gold_label_list: The list of ground truth label of claims
    :type claim_gold_label_list: list
    :param nli_model: The NLI model to calculate coherence metrics (an object of a class inherited from NLIStructure)
    :type nli_model: object

    :ivar rouge: The object to calculate the rouge score
    :vartype rouge: object
    :ivar bertscore: The object to calculate the BERTScore
    :vartype bertscore: object
    :ivar bleu: The object to calculate the bleu score
    :vartype bleu: object        
    '''
    def __init__(self, pred_list = None, target_list= None
        , claim_list= [], claim_gold_label_list=[], nli_model= None):
        
        self.pred_list= pred_list
        self.target_list= target_list
        self.claim_list= claim_list
        self.claim_gold_label_list= claim_gold_label_list
        self.nli_model= nli_model

        self.rouge= None
        self.bertscore= None
        self.bleu= None


    def rouge_score(self):
        ''' This function calculate the rouge score for pred_list regarding target_list.

        :returns: The average rouge score for the list
        :rtype: float
        '''

        if self.rouge is None:
            self.rouge = ROUGEScore()
            
        log("Start calculating ROUGE score ...")
        log(len(self.pred_list), len(self.target_list))
        rouge_result= self.rouge(self.pred_list, self.target_list)
        log(rouge_result)
        return rouge_result


    def bleu_score(self):
        ''' This function calculate the bleu score for pred_list regarding target_list.

        :returns: The average bleu score for the list
        :rtype: float
        '''

        if self.bleu is None:
            self.bleu = BLEUScore()        
        
        bleu_avg= 0
        log("Start calculating BLEU score ...")
        target_count= len(self.target_list)
        # Calculate the average bleu score for all instances in the list
        for index, (pred, target) in enumerate(zip(self.pred_list, self.target_list)):
            if (index+1) % 100 == 0:
                log(f"-------- {index+1}/{target_count} --------")            
            bleu_avg+= self.bleu([pred], [[target]]).item()

        rounded_score = round(bleu_avg / target_count, 4)
        log(f"BLEU Score: {rounded_score}")
        return rounded_score


    def __check_coherence_inputs(func):
        '''
        This is a decorator to check the required inputs of coherence functions.
        '''
        def wrapper(self, *args, **kwargs):

            assert len(self.claim_list) > 0, "Please set the related claim list to the predicted/generated list"
            assert self.nli_model is not None, "Please set the NLI object for inference"

            func_result = func(self, *args, **kwargs)
            return func_result

        return wrapper


    @__check_coherence_inputs
    def SGC(self):
        ''' This function calculates the strong global coherence metric by following the definition of the "Explainable Automated Fact-Checking for Public Health Claims" paper.
        The definition: Every sentence in the generated explanation text must entail the claim.

        :returns: The percentage of instances that satisfy this metric.
        :rtype: float
        '''        
        failed_no= 0
        log("Start calculating SGC score ...")
        target_count= len(self.claim_list)
        for index, (claim, pred) in enumerate(zip(self.claim_list, self.pred_list)):
            if (index+1) % 100 == 0:
                log(f"-------- {index+1}/{target_count} --------")
            
            if not is_farsi(pred):
                failed_no+= 1
                continue
            
            pred_sents_list= sent_tokenize(pred)
            for sent in pred_sents_list:
                if self.nli_model.predict_nli(sent, claim) != NLI_LABEL_ID["entailment"]:
                    failed_no+= 1
                    break
        
        rounded_score = round(1-(failed_no/target_count), 4)
        log(f"SGC Score: {rounded_score}")
        return rounded_score


    @__check_coherence_inputs
    def WGC(self):
        ''' This function calculates the weak global coherence metric by following the definition of the "Explainable Automated Fact-Checking for Public Health Claims" paper.
        The definition: No sentence in the generated explanation text should contradict the claim.

        :returns: The percentage of instances that satisfy this metric.
        :rtype: float
        '''
        assert len(self.claim_gold_label_list) > 0, "Please set the claim's ground truth label list"

        failed_no= 0
        log("Start calculating WGC score ...")
        target_count= len(self.claim_list)
        for index, (claim, pred, claim_label) in enumerate(zip(self.claim_list, self.pred_list, self.claim_gold_label_list)):
            if (index+1) % 100 == 0:
                log(f"-------- {index+1}/{target_count} --------")
            
            if not is_farsi(pred):
                failed_no+= 1
                continue

            pred_sents_list= sent_tokenize(pred)
            for sent in pred_sents_list:
                if self.nli_model.predict_nli(sent, claim) == NLI_LABEL_ID["contradiction"] and claim_label != stance_labels["disagree"]:
                    failed_no+= 1
                    break
        
        rounded_score = round(1-(failed_no/target_count), 4)
        log(f"WGC Score: {rounded_score}")
        return rounded_score


    @__check_coherence_inputs
    def LC(self):
        ''' This function calculates the local coherence metric by following the definition of the "Explainable Automated Fact-Checking for Public Health Claims" paper.
        The definition: Any two sentences in the generated explanation text must not contradict each other.

        :returns: The percentage of instances that satisfy this metric.
        :rtype: float
        '''
        failed_no= 0
        log("Start calculating LC score ...")
        target_count= len(self.pred_list)

        def _locally_coherent(pred):
            pred_sents_list= sent_tokenize(pred)
            for sent1 in pred_sents_list:
                for sent2 in pred_sents_list:
                    if sent1 != sent2 and self.nli_model.predict_nli(sent1, sent2) == NLI_LABEL_ID["contradiction"]:
                        return False

            return True

        for index, pred in enumerate(self.pred_list):
            if (index + 1) % 100 == 0:
                log(f"-------- {index + 1}/{target_count} --------")
            
            if not is_farsi(pred):
                failed_no+= 1
                continue

            failed_no += int(not _locally_coherent(pred, ))

        rounded_score = round(1-(failed_no/target_count), 4)
        log(f"LC Score: {rounded_score}")
        return rounded_score


    def get_all_metrics(self):
        ''' This function calculate all scores to evaluate the pred_list regarding the target_list.

        :returns: The average score for all metrics
        :rtype: dict
        '''

        # return {"rouge": self.rouge_score(), "SGC": self.SGC(), "WGC": self.WGC(), "LC": self.LC(), "bleu": self.bleu_score()}
        # return {"SGC": self.SGC(), "WGC": self.WGC(), "LC": self.LC()}
        return {"rouge": self.rouge_score(), "SGC": self.SGC(), "WGC": self.WGC(), "LC": self.LC()}
                

class NLI(NLIStructure):
    '''
    The NLI object is responsible for obtaining NLI labelof an instance.

    :param model_path: The path of selected model to load
    :type model_path: str

    :ivar _nli_tokenizer: The loaded tokenizer to prevent loading several times
    :vartype _nli_tokenizer: object
    :ivar _nli_model: The loaded NLI model to prevent loading several times
    :vartype _nli_model: object
    :ivar _device: The target device (CPU or GPU) on which the model will be loaded.
    :vartype _device: object
    '''
    def __init__(self, model_path):
            
        self.model_path= model_path

        self._nli_tokenizer= None
        self._nli_model= None
        self._device= torch.device('cpu')


    def predict_nli(self, premise, hypothesis):
        ''' This function returns the NLI label ID with "entailment": 0, "neutral": 1, and "contradiction": 2.

        :param premise: The premise
        :type premise: str
        :param hypothesis: The hypothesis
        :type hypothesis: str

        :returns: The NLI label ("entailment": 0, "neutral": 1, and "contradiction": 2)
        :rtype: int
        '''
        
        return self.__hf_nli_model(premise, hypothesis)


    def __hf_nli_model(self, premise, hypothesis):
        ''' This function returns the NLI label ID with "entailment": 0, "neutral": 1, and "contradiction": 2.
        It uses roberta_large_snli (ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli) transformer to predict the label.

        :param premise: The premise
        :type premise: str
        :param hypothesis: The hypothesis
        :type hypothesis: str

        :returns: The NLI label ("entailment": 0, "neutral": 1, and "contradiction": 2)
        :rtype: int
        '''

        if self._nli_model is None or self._nli_tokenizer is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
                print("The cuda is available.\n")
            else:
                print("The cuda is not available.\n")

            self._nli_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self._device)

        input_ids= self._nli_tokenizer.encode_plus(premise, hypothesis, max_length=200
        , truncation=True, return_tensors="pt")
        transfered_input_ids = {k: v.to(self._device) for k, v in input_ids.items()}
        outputs = self._nli_model(**transfered_input_ids)

        return np.argmax(torch.softmax(outputs[0], dim=1)[0].tolist())


def get_pred_target_colms(file_path, *args):
    ''' This function read a file and returns the content of the selected columns in args.

    :param file_path: The path to the file to report metrics for it
    :type file_path: str
    :param *args: The title of target columns to get their contents as a list
    :type *args: str

    :returns: The content of selected columns
    :rtype: tuple
    '''

    path = Path(file_path)
    assert path.is_file(), f"Please enter a correct path to a csv file."
    target_file_df= pd.read_excel(file_path)
    target_file_df= target_file_df.fillna('')
    lst_result=[]
    
    for arg in args:
        lst_result.append(target_file_df[arg].tolist())

    return tuple(lst_result)


def report_nle_metrics(file_path, pred_col_title, target_col_title, metrics_obj, target_metric):
    ''' This function read a file and report evaluation metrics for the generated explanation in the file.

    :param file_path: The path to the file to report metrics for it
    :type file_path: str
    :param pred_col_title: The title of the prediction column to report metrics for it
    :type pred_col_title: str
    :param target_col_title: The title of target (ground truth) column to report metrics regarding it
    :type target_col_title: str
    :param metrics_obj: An instance of NLEMetrics class to obtain metrics
    :type metrics_obj: object
    :param target_metric: The target metric you want to calculate
    :type target_metric: str

    :returns: The calculated metrics
    :rtype: dict
    '''
    
    metrics_obj.pred_list, metrics_obj.target_list, metrics_obj.claim_list, metrics_obj.claim_gold_label_list = get_pred_target_colms(file_path, pred_col_title, target_col_title, "claim", "stance")

    metric_fun_map = {'all': metrics_obj.get_all_metrics
                        , 'rouge': metrics_obj.rouge_score
                        , 'SGC': metrics_obj.SGC
                        , 'WGC': metrics_obj.WGC
                        , 'LC': metrics_obj.LC
                        , 'bleu': metrics_obj.bleu_score}

    target_metric_result= metric_fun_map[target_metric]()
    log(target_metric_result)
    log("-"* 100)
    log(f"{round(target_metric_result['rouge']['rouge1_fmeasure'].item() * 100, 1)} & {round(target_metric_result['rouge']['rouge2_fmeasure'].item() * 100, 1)} & {round(target_metric_result['rouge']['rougeL_fmeasure'].item() * 100, 1)} & {round(target_metric_result['SGC'] * 100, 1)} & {round(target_metric_result['WGC'] * 100, 1)} & {round(target_metric_result['LC'] * 100, 1)}")
    return target_metric_result


def report_stance_metrics(file_path, pred_col_title, target_col_title, stance_average_method):
    ''' This function read a file and report evaluation metrics for stance classification task.

    :param file_path: The path to the file to report metrics for it
    :type file_path: str
    :param pred_col_title: The title of the prediction column to report metrics for it
    :type pred_col_title: str
    :param target_col_title: The title of target (ground truth) column to report metrics regarding it
    :type target_col_title: str
    :param stance_average_method: The average method fot reporting recal, precision, and F1.
    :type stance_average_method: str
    
    :returns: The calculated metrics
    :rtype: dict
    '''

    pred_list, target_list = get_pred_target_colms(file_path, pred_col_title, target_col_title)

    labels= [0,1,2,3]

    log("Classes: ", labels)

    metrics_result= {"acc": accuracy_score(y_pred=pred_list, y_true=target_list)
        , "pre": precision_score(y_pred=pred_list, y_true=target_list, average=stance_average_method, labels=labels)
        , "rec": recall_score(y_pred=pred_list, y_true=target_list, average=stance_average_method, labels=labels)
        , "f1": f1_score(y_pred=pred_list, y_true=target_list, average=stance_average_method, labels=labels)
        , "confmat": confusion_matrix(y_pred=pred_list, y_true=target_list, labels=labels)}

    
    log(f"{round(metrics_result['pre'] * 100, 1)} & {round(metrics_result['rec'] * 100, 1)} & {round(metrics_result['f1'] * 100, 1)} & {round(metrics_result['acc'] * 100, 1)}")

    return metrics_result


def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
       
    parser.add_argument("-pred_col_title", "--pred_col_title", help = "The title of the prediction column to report metrics for it"
        , default='gen_explanation', type= str)
    parser.add_argument("-target_col_title", "--target_col_title", help = "The title of target (ground truth) column to report metrics regarding it"
        , default='evidence', type= str)
    parser.add_argument("-type", "--type"
        , help = "Report metrics for a single file or all files in a directory", default="directory"
        , choices=['file', 'directory'], type= str)
    parser.add_argument("-target_path", "--target_path"
        , help = "The path to the file or directory to report metrics for them", default="data/prompts_output/"
        , type= str)
    parser.add_argument("-task_type", "--task_type"
        , help = "Report metrics for the stance prediction, the explanation generation or the joint model", default="stance"
        , choices=['explanation', 'stance'], type= str)
    parser.add_argument("-stance_average_method", "--stance_average_method"
        , help = "The average method fot reporting recal, precision, and F1.", default="macro"
        , choices=['macro', 'micro', 'weighted'], type= str)
    parser.add_argument("-nli_model_path", "--nli_model_path"
        , help = " The path of selected model to calculate the SGC and WGC scores."
        , default="parsi-ai-nlpclass/ParsBERT-nli-FarsTail-FarSick", type= str)
    parser.add_argument("-target_metric", "--target_metric"
        , help = "The target metric you want to calculate. It is only for explanation!", default="all"
        , choices=['all', 'rouge', 'SGC', 'WGC', 'LC', 'bleu'], type= str)        

    # Read arguments from command line
    args = parser.parse_args()
    
    nle_metrics= NLEMetrics()
    nle_metrics.nli_model= NLI(args.nli_model_path)

    files = []

    log("Input arguments for calculating metric(s): ", vars(args))

    if args.type == "file":
        files.append(args.target_path)
    else:
        # csv files in the path
        files = glob.glob(args.target_path + "/*.csv")

    for file_name in files:
        log("-"*50)
        log(f"Calculating metrics for {file_name}:")
        
        metric_results= None
        #  save results in a csv file
        save_path= file_name.replace(".csv", "").replace(".xlsx", "")+"/"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        if args.task_type=="stance":
            metric_results= report_stance_metrics(file_name, args.pred_col_title, args.target_col_title, args.stance_average_method)
            result_file_name= f"{save_path}{args.task_type}_{args.stance_average_method}_results.csv"
            log(metric_results)
        elif args.task_type=="explanation":
            metric_results= report_nle_metrics(file_name, args.pred_col_title, args.target_col_title, nle_metrics, args.target_metric)
            result_file_name= f"{save_path}{args.task_type}_{args.target_metric}_results.csv"
        else: # joint task
            pass
        
        df = pd.DataFrame.from_dict([metric_results])
        df.to_csv(result_file_name)


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here

    # try:
    main()
    # except Exception as err:
    #     log(f"Unexpected error: {err}, type: {type(err)}")