python rag_inference.py -model_type OpenAI -model_id gpt-4o -prompt_type zero -test_set test_set_final.xlsx -test_samples_no 1440 -demon_samples_no 0 -temperature 1 -embedding_model sentence-transformers/all-MiniLM-L12-v2 -generation_max_token 150 -similar_chunks_no 2 -similarity_score_threshold 0.2