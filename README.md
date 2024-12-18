# FarExStance: Explainable Stance Detection for Farsi  

**Note:** Updates to this repository are coming soon!  

This repository contains the data and code for the paper [FarExStance: Explainable Stance Detection for Farsi](https://coling2025.org/program/main_conference_papers/#long-papers), which will be presented at the 31st International Conference on Computational Linguistics (COLING 2025).  

## Introduction  
_FarExStance_ introduces the first and largest claim-based explainable stance detection dataset for Farsi. This dataset enables new research in stance detection by providing high-quality annotations and supporting evidence.  

We conducted extensive experiments to establish baseline performance using various multilingual open-source and proprietary models, including small and large language models, retrieval-augmented generation (RAG), and parameter-efficient fine-tuning methods. Both automatic and human evaluations were performed to analyze the strengths and limitations of these approaches.  

Our dataset, curated with manually labeled instances and supporting evidence, is publicly available to facilitate further research in areas such as explainable NLP and the social media domain.  

## Data  
The dataset is organized in the `data/` directory:  

- A list of 130+ Farsi news agency websites, used to collect political, economic, and sports news over six months, is available in `news_agency_websites_list.json`.  
- Training, development, and test sets for the _article2claim_ and social media domains can be found in the `data/b2c/` directory.  
- Training, development, and test sets for the _head2claim_ task are in the `data/h2c/` directory.  

## Reproducing the Experiments  
To reproduce the experiments from our paper:  

1. Use the provided `*.sh` scripts with the appropriate arguments. For example:  
   - Running `huggingface_exp.sh` reproduces the zero-shot and few-shot results of the explainable stance detection task on the test set using models like [Command-R-32B](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024) and [Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B).  
   
2. The generated results (predicted stance and explanation) for all instances in the test set for each LLM are available in the corresponding directories under `data/`.  

3. A detailed explanation of each argument can be found in the [rag_inference.py file](https://github.com/Dadmatech/FarExStance/blob/main/rag_inference.py).  
