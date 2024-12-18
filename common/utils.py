
def get_my_root_path_on_server():
    return "/home/jovyan/Zarharan/"


def get_transformers_root_path():
    return (get_my_root_path_on_server() 
        + "transformers/")


def get_sentence_similarity_model_path():
    return "/home/jovyan/sajadi/etezadi/SentenceSimilarity/models/0_Transformer"


def get_ner_model_path():
    return (get_transformers_root_path() + "parsbert_ner/model")


def get_ner_tokenizer_path():
    return (get_transformers_root_path() + "parsbert_ner/tokenizer")    


def get_en_ner_model_path(): # "dslim/bert-base-NER"
    return (get_transformers_root_path() + "en_ner/model")


def get_en_ner_tokenizer_path(): # "dslim/bert-base-NER"
    return (get_transformers_root_path() + "en_ner/tokenizer")


def get_en_sentence_similarity_model_path(): # "sentence-transformers/stsb-distilbert-base"
    return (get_transformers_root_path() + "en_stsb")
    