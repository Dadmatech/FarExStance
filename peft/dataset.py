from functools import partial
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import sys
import os

# Add the directory containing test.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))

from preprocessor import *


def read_dataset(file_path, validate_mode= False):
    """
    Reading the target dataset method

    :param file_path: The dataset file path
    :param validate_mode: Prepare dataset with prompts for evaluating or training
    """

    Id2Label= ["Disagree", "Agree", "Discuss", "Unrelated"]
    if ".xlsx" in file_path:
        target_file_df= pd.read_excel(file_path)
    else:
        target_file_df= pd.read_csv(file_path)

    target_file_df= target_file_df.filter(items=['news_id', 'claim_id', 'claim', 'content', 'stance', 'evidence'])

    # Remove instances that include null values
    target_file_df = target_file_df.dropna()

    lst_texts= []

    # Add prompt to all rows
    prompt= """### Context: {0}\n\n### Claim: {1}\n\n### Response:\n{2}{3}"""
    
    preprocessor= Preprocessor()

    for index, row in target_file_df.iterrows():
        gold_stance= Id2Label[row["stance"]]
        gold_evidence= "\n" + row["evidence"]
        if gold_stance== "Unrelated":
            gold_evidence= "\n" + "ادعای وارد شده در متن گزارش نشده است."

        if validate_mode:
            gold_stance= ""
            gold_evidence= ""
        lst_texts.append(prompt.format(preprocessor.clean_text(row["content"]), row["claim"], gold_stance, gold_evidence))

    dataset= Dataset.from_pandas(pd.DataFrame(data={"text": lst_texts}))

    print(f'Number of prompts: {len(dataset)}')
    print(f'Column names are: {dataset.column_names}')

    return dataset


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes dataset batch

    :param batch: Dataset batch
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        batch["text"],
        max_length = max_length,
        truncation = True,
        padding=True,
    )


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str, remove_columns:list):
    """
    Tokenizes dataset for fine-tuning

    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param seed: Random seed for reproducibility
    :param dataset (str): Instruction dataset
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched = True,
        remove_columns = remove_columns
    )

    # Filter out samples that have "input_ids" exceeding "max_length"
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= max_length)

    print("----------- instances after filtering out long samples: ", len(dataset))

    # Shuffle dataset
    dataset = dataset.shuffle(seed = seed)

    return dataset
