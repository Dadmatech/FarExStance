from dotenv import dotenv_values
import os
import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
import torch
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from pathlib import Path
import argparse
from common.preprocessor import Preprocessor
from peft import AutoPeftModelForCausalLM
import anthropic
import time
from openai import RateLimitError
from openai import OpenAI
import backoff


# take Hugging face and Open AI APIs' secret keys from .env file.
secret_keys = dotenv_values(".env")
HF_TOKEN= secret_keys["HF_TOKEN"]
os.environ['OPENAI_API_KEY'] = secret_keys["OAI_API_KEY"]
os.environ['ANTHROPIC_API_KEY']=secret_keys["CLAUDE_KEY"]


labels= {
    "disagree": 0,
    "agree": 1,
    "discuss": 2,
    "unrelated": 3,
    "other": 4
}
Id2Label= ["Disagree", "Agree", "Discuss", "Unrelated"]
words_to_remove_from_generation= ["Response:", "Response", "response:", "response", "[", "]", "{", "}", "....", "Category:", "Category", "category:", "category", "موافق\n", "موافقت\n", "توافق\n", "مخالف\n", "بحث\n", "نامرتبط\n", "دیسکاس\n", "بیربط\n", "بی ربط\n"]


prompt_templates= {
    "first_attempt": """Classify the stance of the following context against the claim as Agree, Disagree, Discuss, or Unrelated. And explain your reasoning. Format the output as a list, first item is the stance label and the second item is your explanation in Farsi.
        Context: {context}

        Claim: {claim}
        {discriminant}\n
        """,
    "second_attempt": """Analyze the claim's relationship to the given context. Categorize it as:
        - Agree: Context unequivocally supports the claim's truth.
        - Disagree: Context unequivocally refutes the claim's truth.
        - Discuss: Context offers neutral information or reports the claim without evaluating its veracity.
        - Unrelated: Claim is not addressed in the context.
        
        Output format:
        [Category]
        [Explanation in Farsi]
        
        Provide only the category from Agree, Disagree, Discuss, and Unrelated in English, and Farsi explanation based only on the context. No additional text!!!!.

        Context: {context}

        Claim: {claim}
        {discriminant}\n
        """,
    "third_attempt": """### Instruction:\nUse the Task below and the Input given to write the Response, which is a stance label prediction that can solve the Task.
            \n\n### Task:\nCategorize the stance of the following context against the claim as:
            - Agree: Context unequivocally supports the claim's truth.
            - Disagree: Context unequivocally refutes the claim's truth.
            - Discuss: Context offers neutral information or reports the claim without evaluating its veracity, or context missed some details in the claim.
            - Unrelated: Claim is not addressed in the context.
            
            Output format should be:
            [Category]
            [Explanation in Farsi]
            
            Provide only the category and Farsi explanation based only on the context. No additional text!!!!.
            \n\n### Input:\nContext: {context}\nClaim: {claim}\n\n{discriminant} Response:\n""",
    "fourth_attempt": """### Instruction:\nUse the Task below and the Input given to write the Response, which is a stance label prediction that can solve the Task.
            \n\n### Task:\nCategorize the stance of the following context against the claim as:
            - Agree: Context unequivocally supports the claim's truth.
            - Disagree: Context unequivocally refutes the claim's truth.
            - Discuss: Context offers neutral information or reports the claim without evaluating its veracity, or context missed some details in the claim.
            - Unrelated: Claim is not addressed in the context.
            
            Output format should be:
            [Category]
            [Explanation in Farsi]
            
            Provide only the category and Farsi explanation based only on the context. No additional text!!!!. Think step-by-step before you write the response.
            \n\n### Input:\nContext: {context}\nClaim: {claim}\n\n{discriminant} Response:\n""",            
    "fewshot_template": """### Instruction:\nUse provided examples below to learn more about stance detection and explanation.\n\n{examples}\n\n### Instruction:\nNow, use the Task below and the Input given to write the Response, which is a stance label prediction that can solve the Task.
            \n\n### Task:\nCategorize the stance of the following context against the claim as:
            - Agree: Context unequivocally supports the claim's truth.
            - Disagree: Context unequivocally refutes the claim's truth.
            - Discuss: Context offers neutral information or reports the claim without evaluating its veracity, or context missed some details in the claim.
            - Unrelated: Claim is not addressed in the context.
            
            Output format should be:
            [Category]
            [Explanation in Farsi]
            
            Provide only the category and Farsi explanation based only on the context. No additional text!!!!.
            \n\n### Input:\nContext: {context}\nClaim: {claim}\n\n{discriminant} Response:\n""",            
    "fewshot_examples": """### Example {index}:\nCategorize the stance of the following context against the claim as:
            - Agree: Context unequivocally supports the claim's truth.
            - Disagree: Context unequivocally refutes the claim's truth.
            - Discuss: Context offers neutral information or reports the claim without evaluating its veracity, or context missed some details in the claim.
            - Unrelated: Claim is not addressed in the context.
            
            Output format should be:
            [Category]
            [Explanation in Farsi]
            
            Provide only the category and Farsi explanation based only on the context. No additional text!!!!.
            \n\n### Input:\nContext: {context}\nClaim: {claim}\n\nResponse:\n{stance}\n{explanation}""",
    "peft_template": """### Context: {context}\n\n### Claim: {claim}\n\n### Response:\n"""
}


system_message= """You are a helpful assistant that predicts the stance of a context against a claim and explains the reason for your prediction by considering the context.
        Instructions: {0}Categorize the stance of the following context against the claim as:
            - Agree: Context unequivocally supports the claim's truth.
            - Disagree: Context unequivocally refutes the claim's truth.
            - Discuss: Context offers neutral information or reports the claim without evaluating its veracity, or context missed some details in the claim.
            - Unrelated: Claim is not addressed in the context.
            
            Output format should be:
            [Category]
            [Explanation in Farsi]
            
            Provide only the category and Farsi explanation based only on the context. Think step-by-step before you write the response."""

CHAT_EXTRA_DESC= {
    "zero": "",
    "few": "Use the provided examples to learn more about the stance detection and providing explanation."
}


class RAGInference():

    def __init__(self, args, few_shot_samples_fname):
        self.few_shot_set= None
        self.few_shot_samples_fname= few_shot_samples_fname
        self.args= args
        self.few_shot_examples= ""

        if args.prompt_type== "few":
            # assert args.demon_set, "Enter the demonstration set name by using -demon_set argument!"
            self.few_shot_set= pd.read_excel(f"data/b2c/{self.few_shot_samples_fname}.xlsx")    

        print(f"args.load_in_4bit: {args.load_in_4bit}, args.load_in_8bit: {args.load_in_8bit}")

        self.target_df= pd.read_excel(f"data/b2c/{args.test_set}")

        self.retriever_embedding_model = HuggingFaceEmbeddings(
            model_name = args.embedding_model,
            encode_kwargs = {'normalize_embeddings':True}
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.target_df = self.target_df.sample(args.test_samples_no, random_state= args.seed)
        self.target_df= self.target_df.fillna("")
        self.preprocessor= Preprocessor()


    def __clean_llm_result(self, raw_result):
        '''
        This function extracts and returns the stance label and explanation from raw_result

        :param raw_result: The input text
        :type raw_result: str

        :returns: Extracted stance label and explanation
        :rtype: tuple
        '''

        if raw_result.strip() == "":
            return labels["other"], "No Explanation"

        def _get_related_section(input_text, segmentor_keyword="### Output:"):
            # remove the curly braces from the string
            input_text = input_text.strip('[{}]').split(segmentor_keyword)
            result_section= input_text[0]

            if len(input_text)>1:
                for label in Id2Label:
                    if label in input_text[1] or label.lower() in input_text[1]:
                        result_section= input_text[1].split("###")[0]
                        break
            return result_section
        
        def _clean_explanation(exp):
            for val in words_to_remove_from_generation:
                exp= exp.replace(val, "")
            for label in Id2Label:
                exp= exp.replace(label, "").replace(label.lower(), "")

            return exp.strip()

        result_section= raw_result.strip('[{}]').split("###")[0]

        for segmentor in ["### Answer:", "### Output:", "### Response:", "### Stance:", "### Human Response:"]:
            if segmentor in raw_result:
                result_section= _get_related_section(raw_result, segmentor)
                break

        if "### Agree" in raw_result:
            return labels["agree"], _clean_explanation(raw_result.split("### Agree")[1])
        elif "### Disagree" in raw_result:
            return labels["disagree"], _clean_explanation(raw_result.split("### Disagree")[1])
        elif ("### Discuss" in raw_result):
            return labels["discuss"], _clean_explanation(raw_result.split("### Discuss")[1])
        elif ("### Unrelated" in raw_result):
            return labels["unrelated"], _clean_explanation(raw_result.split("### Unrelated")[1])                     
        elif ("موافق\n" in raw_result) or ("موافقت\n" in raw_result) or ("توافق\n" in raw_result):
            return labels["agree"], _clean_explanation(raw_result)
        elif ("مخالف\n" in raw_result):
            return labels["disagree"], _clean_explanation(raw_result)
        elif ("بحث\n" in raw_result) or ("دیسکاس\n" in raw_result):
            return labels["discuss"], _clean_explanation(raw_result)
        elif ("نامرتبط\n" in raw_result) or ("بیربط\n" in raw_result) or ("بی ربط\n" in raw_result):
            return labels["unrelated"], _clean_explanation(raw_result)                     
        
        all_lines = [line.strip().lower().replace("#", "") for line in result_section.split("\n") if len(line.strip().replace("#", ""))>1]

        if len(all_lines) == 0:
            return labels["other"], "No Explanation"
        
        clean_result= " ".join(all_lines)
        predicted_label= "other"
        for label in Id2Label:
            if label in clean_result or label.lower() in clean_result:
                predicted_label= label
                break
        
        clean_lines = [line.replace((label.lower()+":"), "").replace((label+":"), "").replace(label.lower(), "").replace(label, "") for line in all_lines]
        explanation= " ".join([line.strip() for line in clean_lines if len(line.strip())>3])

        return labels[predicted_label.lower()], _clean_explanation(explanation)


    def __get_fewshot_examples(self, shuffle= False):
        few_shot_set = self.few_shot_set

        if shuffle:
            few_shot_set = few_shot_set.sample(frac = 1)

        shot_examples= []
        for index in range(len(few_shot_set)):
            
            prompt = """\nContext: {0}\nClaim: {1}\n{2}\n{3}"""
            shot_examples.append(prompt.format(few_shot_set.iloc[index].matched_content
                                ,few_shot_set.iloc[index].claim
                                ,Id2Label[few_shot_set.iloc[index].stance]
                                ,few_shot_set.iloc[index].evidence))
        self.few_shot_examples= "\n\n".join(shot_examples) + "\n"
        return self.few_shot_examples


    def __get_related_chunks(self, org_context, claim, collection_name):
        # Split
        text_splitter = SemanticChunker(self.retriever_embedding_model, breakpoint_threshold_type="percentile")
        # Make splits
        splits = text_splitter.create_documents([self.preprocessor.clean_text(org_context)])

        # Embedding and Vector DataBase
        vectorstore = Chroma.from_documents(collection_name= collection_name
                                            , documents=splits, embedding=self.retriever_embedding_model)

        # Retriever
        retriever = vectorstore.as_retriever(#search_type="similarity_score_threshold"
                                                search_kwargs={"k" : self.args.similar_chunks_no})
                                                #, search_kwargs={"score_threshold": args.similarity_score_threshold, "k" : args.similar_chunks_no})
        matched_docs = retriever.invoke(claim)
        return matched_docs


    def __huggingface_based_models(self):
        bnb_config= None
        
        if self.args.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = torch.bfloat16,
            )
        elif self.args.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit = True,
                llm_int8_threshold = 6.0
            )

        tokenizer = AutoTokenizer.from_pretrained(self.args.model_id, token=HF_TOKEN)

        if self.args.prompt_type=="peft":
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.args.model_id,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                quantization_config = bnb_config,
                token= HF_TOKEN
            )

            model.eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(self.args.model_id, token=HF_TOKEN, device_map="auto"
                                                    , temperature=self.args.temperature
                                                    , do_sample=True
                                                    , quantization_config = bnb_config)
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=self.args.generation_max_token
        )
        hf_model= HuggingFacePipeline(pipeline=pipe)

        lst_result= []
        lst_prompts= []
        lst_labels= []
        lst_explanation= []
        discriminant= "#####"
        
        for index in range(len(self.target_df)):
            
            matched_docs= self.__get_related_chunks(org_context= self.target_df.iloc[index].content
                                                    , claim= self.target_df.iloc[index].claim
                                            , collection_name= f"{self.target_df.iloc[index].claim_id}_{self.target_df.iloc[index].news_id}")

            prompt = ChatPromptTemplate.from_template(prompt_templates[self.args.prompt_template])

            # Chain
            chain = prompt | hf_model

            # Run
            inputs_dict= {"context":"\n".join([doc.page_content for doc in matched_docs]),"claim":self.preprocessor.clean_text(self.target_df.iloc[index].claim), "discriminant":discriminant}
            if self.args.prompt_type=="few":
                inputs_dict["examples"]= self.__get_fewshot_examples(shuffle= False)

            result= chain.invoke(inputs_dict)
            # print(f"Result:\n{result}")

            result= result.strip()
            if result == "":
                print(f"Empty result")
                lst_result.append("")
                lst_prompts.append("")
                lst_labels.append(labels["other"])
                lst_explanation.append("")
            else:
                output= result.split(discriminant)
                if len(output)<2:
                    lst_result.append("")
                    lst_prompts.append("")
                    lst_labels.append(labels["other"])
                    lst_explanation.append("")
                else:
                    raw_llm_result= output[1].split("Human:")[0]
                    lst_result.append(raw_llm_result)
                    lst_prompts.append(output[0])
                    pred_label, explanation= self.__clean_llm_result(raw_llm_result)
                    lst_labels.append(pred_label)
                    lst_explanation.append(explanation)

        
        self.target_df["raw_llm_result"]= lst_result
        self.target_df["prompt"]= lst_prompts
        self.target_df["pred_stance"]= lst_labels
        self.target_df["gen_explanation"]= lst_explanation

        return self.target_df


    @backoff.on_exception(backoff.expo, RateLimitError)
    def __openai_chat_query(self, s_message, u_message):
        ''' This function send a query to open ai for using Chat-based models.

        :param prompt: The target prompt
        :type prompt: str

        :returns: The generated message
        :rtype: str
        '''
        
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        response= client.chat.completions.create(
            messages=[
                {"role": "system", "content": s_message},
                {
                    "role": "user",
                    "content": u_message,
                }
            ],
            model= self.args.model_id, temperature= self.args.temperature , max_tokens= self.args.generation_max_token, top_p=1
            , frequency_penalty=0, presence_penalty=0
        )

        return response.choices[0].message.content


    def __openai_based_models(self):
        lst_result= []
        lst_prompts= []
        lst_labels= []
        lst_explanation= []
        
        len_target_df= len(self.target_df)
        detailed_dir= f"data/{self.args.model_id}/detailed/{self.args.prompt_type}/"
        for index in range(len_target_df):
            save_file_path= f"{detailed_dir}{self.target_df.iloc[index].claim_id}_{self.target_df.iloc[index].news_id}.xlsx"
            if os.path.exists(save_file_path):
                print(f"Skip!!! news_id:{self.target_df.iloc[index].news_id}, claim_id:{self.target_df.iloc[index].claim_id}")
                exp_result= pd.read_excel(save_file_path)
                lst_result.append(exp_result.iloc[index].raw_llm_result)
                lst_prompts.append(exp_result.iloc[index].prompt)
                lst_labels.append(exp_result.iloc[index].pred_stance)
                lst_explanation.append(exp_result.iloc[index].gen_explanation)
                continue

            time.sleep(1)
            result= ""
            prompt= ""
            pred_label= labels["other"]
            explanation= ""

            try:
                matched_docs= self.__get_related_chunks(org_context= self.target_df.iloc[index].content
                                                        , claim= self.target_df.iloc[index].claim
                                                , collection_name= f"{self.target_df.iloc[index].claim_id}_{self.target_df.iloc[index].news_id}")

                s_message= system_message.format((CHAT_EXTRA_DESC[self.args.prompt_type]) + self.few_shot_examples)
                u_message= """Context:{0}\nClaim:{1}""".format('\n'.join([doc.page_content for doc in matched_docs]), self.preprocessor.clean_text(self.target_df.iloc[index].claim))

                result= self.__openai_chat_query(s_message, u_message)
                prompt= s_message + "\n" + u_message

                pred_label, explanation= self.__clean_llm_result(result)

                result_df= pd.DataFrame.from_dict([{"claim_id": self.target_df.iloc[index].claim_id
                                                    , "news_id": self.target_df.iloc[index].news_id
                                                    , "raw_llm_result": result
                                                    , "prompt": prompt
                                                    , "pred_stance": pred_label
                                                    , "gen_explanation": explanation}])
                # save results in a csv file
                Path(detailed_dir).mkdir(parents=True, exist_ok=True)
                result_df.to_excel(save_file_path)
            except:
                print("EXCEPTION !!!!","claim_id:" , self.target_df.iloc[index].claim_id, "news_id:" , self.target_df.iloc[index].news_id)
            
            lst_result.append(result)
            lst_prompts.append(prompt)
            lst_labels.append(pred_label)
            lst_explanation.append(explanation)

            print(f"End of {index + 1} / {len_target_df}")

        self.target_df["raw_llm_result"]= lst_result
        self.target_df["prompt"]= lst_prompts
        self.target_df["pred_stance"]= lst_labels
        self.target_df["gen_explanation"]= lst_explanation
        return self.target_df


    def __claude_based_models(self):
        lst_result= []
        lst_prompts= []
        lst_labels= []
        lst_explanation= []
        client = anthropic.Anthropic()
        detailed_dir= f"data/{self.args.model_id}/detailed/{self.args.prompt_type}/"

        for index in range(len(self.target_df)):

            # Check the result for each instance, if we have the result for the instance and prompted it, we skip that instance!
            save_file_path= f"{detailed_dir}{self.target_df.iloc[index].claim_id}_{self.target_df.iloc[index].news_id}.xlsx"
            if os.path.exists(save_file_path):
                print(f"Skip!!! news_id:{self.target_df.iloc[index].news_id}, claim_id:{self.target_df.iloc[index].claim_id}")
                exp_result= pd.read_excel(save_file_path)
                lst_result.append(exp_result.iloc[index].raw_llm_result)
                lst_prompts.append(exp_result.iloc[index].prompt)
                lst_labels.append(exp_result.iloc[index].pred_stance)
                lst_explanation.append(exp_result.iloc[index].gen_explanation)
                continue

            time.sleep(1)
            result= ""
            prompt= ""
            pred_label= labels["other"]
            explanation= ""

            try:
                matched_docs= self.__get_related_chunks(org_context= self.target_df.iloc[index].content
                                                        , claim= self.target_df.iloc[index].claim
                                                , collection_name= f"{self.target_df.iloc[index].claim_id}_{self.target_df.iloc[index].news_id}")

                s_message= system_message.format((CHAT_EXTRA_DESC[self.args.prompt_type]) + self.few_shot_examples)
                u_message= """Context:{0}\nClaim:{1}""".format('\n'.join([doc.page_content for doc in matched_docs]), self.preprocessor.clean_text(self.target_df.iloc[index].claim))

                message = client.messages.create(
                    model= self.args.model_id,
                    max_tokens=self.args.generation_max_token,
                    temperature= self.args.temperature,
                    system= s_message,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": u_message
                                }
                            ]
                        }
                    ]
                )

                result= message.content[0].text.strip()
                prompt= s_message + "\n" + u_message

                pred_label, explanation= self.__clean_llm_result(result)

                result_df= pd.DataFrame.from_dict([{"claim_id": self.target_df.iloc[index].claim_id
                                                    , "news_id": self.target_df.iloc[index].news_id
                                                    , "raw_llm_result": result
                                                    , "prompt": prompt
                                                    , "pred_stance": pred_label
                                                    , "gen_explanation": explanation}])
                
                # save results in a csv file
                Path(detailed_dir).mkdir(parents=True, exist_ok=True)
                result_df.to_excel(save_file_path)

            except:
                print("EXCEPTION !!!!","claim_id:" , self.target_df.iloc[index].claim_id, "news_id:" , self.target_df.iloc[index].news_id)
            
            lst_result.append(result)
            lst_prompts.append(prompt)
            lst_labels.append(pred_label)
            lst_explanation.append(explanation)

            print(f"End of index: {index + 1}")

        self.target_df["raw_llm_result"]= lst_result
        self.target_df["prompt"]= lst_prompts
        self.target_df["pred_stance"]= lst_labels
        self.target_df["gen_explanation"]= lst_explanation
        return self.target_df

    
    def get_results(self):
        if self.args.model_type == 'OpenAI':
            return self.__openai_based_models()
        elif self.args.model_type == 'HuggingFace':
            return self.__huggingface_based_models()
        elif self.args.model_type == 'Claude':
            return self.__claude_based_models()


def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-test_samples_no", "--test_samples_no", help = "The number of samples for test section of prompt paradigm"
    , default=1, type= int)

    # 0 mean zero-shot experiment
    parser.add_argument("-demon_samples_no", "--demon_samples_no", help = "The number of samples for demonstration section of prompt paradigm"
    , default=1, type= int)

    parser.add_argument("-demon_set", "--demon_set"
    , help = "The target dataset name to select the demonstration instances from"
    , type=str, default='dev_set_final.xlsx')

    parser.add_argument("-test_set", "--test_set"
    , help = "The target dataset name to select the test instances from"
    , type=str, default='test_set_final.xlsx')

    parser.add_argument("-seed", "--seed", help = "seed for random function. Pass None for select different instances randomly."
    , default=313, type= int)

    parser.add_argument("-prompt_template", "--prompt_template", help = "The target template to create prompt"
    , default='first_attempt', choices=['first_attempt', 'second_attempt', 'third_attempt', 'fewshot_template', "peft_template"])

    parser.add_argument("-model_type", "--model_type", help = "The type of model which can include OpenAI, Claude, and HuggingFace models."
    , default='HuggingFace', choices=['OpenAI', 'HuggingFace', 'Claude'])    

    parser.add_argument("-model_id", "--model_id", help = "c4ai-command-r-08-2024 or Llama 3.1"
    , default= "CohereForAI/c4ai-command-r-08-2024", choices=["CohereForAI/c4ai-command-r-08-2024",'meta-llama/Meta-Llama-3.1-70B', 'claude-3-5-sonnet-20240620', 'gpt-4o', 'CohereForAI/aya-23-8B', 'peft/results/CohereForAI/aya-23-8B/10_rank16_dropout0.5_dropout16/checkpoint-15519'])

    parser.add_argument("-embedding_model", "--embedding_model", help = "Sentence transformers for embedding"
    , default= "BAAI/bge-base-en-v1.5", choices=["BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L12-v2"])    

    parser.add_argument("-generation_max_token", "--generation_max_token", help = "The max number of tokens for the generated text."
    , default=300, type= int)

    parser.add_argument("-temperature", "--temperature", help = "To set the randomness of generated text."
    , default=0.1, type= float)    

    parser.add_argument("-similar_chunks_no", "--similar_chunks_no", help = "The max number of chunks for choosing by similarity as context."
    , default=3, type= int)

    parser.add_argument("-similarity_score_threshold", "--similarity_score_threshold", help = "The min score_threshold for similarity search."
    , default=0.4, type= float)
    
    parser.add_argument("-load_in_4bit", "--load_in_4bit", help = "True or False"
    , default=False, type= bool)

    parser.add_argument("-load_in_8bit", "--load_in_8bit", help = "True or False"
    , default=False, type= bool)    
    
    parser.add_argument("-prompt_type", "--prompt_type", help = "zero shot or few shot"
    , default='zero', choices=['zero', 'few', 'peft'])

    # Read arguments from command line
    args = parser.parse_args()

    assert args.test_set, "At least enter the test set name by using -test_set argument!"

    few_shot_samples_fname= "few_shot_samples_revised_unrelated"

    rag_inference_obj = RAGInference(args=args, few_shot_samples_fname= few_shot_samples_fname)

    target_df= rag_inference_obj.get_results()

    target_path= f"data/{args.model_id}/{args.prompt_type}_shot/"

    # save results in a csv file
    Path(target_path).mkdir(parents=True, exist_ok=True)

    if args.model_type == 'OpenAI' or args.model_type == 'Claude':
        result_file_path= f"{target_path}{args.test_samples_no}_result_on_{args.test_set}"
    elif args.model_type == 'HuggingFace':
        result_file_path= f"{target_path}{args.test_samples_no}_{args.prompt_template}_result_on_{args.test_set}"
    
    target_df.to_excel(result_file_path)
    print(f"Done, check the results in {result_file_path}")


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    # try:
    main()
    # except Exception as err:
    #     log(f"Unexpected error: {err}, type: {type(err)}")