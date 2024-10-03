import os
import time
import torch
import logging
from utils import load_data, evaluate
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

def main():
    start_time = time.time()
    
    torch.cuda.set_device(2)
    
    # Configure logging once outside the loop
    logging.basicConfig(level=logging.INFO)
    
    # List of datasets to process
    datasets = [
        'trec-covid',
        'fiqa',
        'nfcorpus',
        'webis-touche2020',
        'dbpedia-entity'
    ]
    
    # Iterate over each dataset
    for dataset_name in datasets:    
        result_file = f"/home/intern/Leon_Kuo/Paper/Evaluation_Results/result-rawQ-{dataset_name}.json"
        
        # Check if result file exists and is non-empty
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            logging.info(f"Skipping dataset: {dataset_name}. Results already exist and file is non-empty.")
        else:
            logging.info(f"Processing dataset: {dataset_name}")

            # Load data
            corpus, queries, qrels = load_data(dataset_name)


            ############
            # Initialize the retriever here
            model = DRES(models.SentenceBERT("facebook/contriever"), batch_size=64)
            retriever = EvaluateRetrieval(model, score_function="dot")
            
            """
            # Initialize the retriever here
            model = DRES(models.SentenceBERT("intfloat/e5-small-v2"), batch_size=64)
            retriever = EvaluateRetrieval(model, score_function="cos_sim")
            """
            ############
            
            
            retrieval_result = retriever.retrieve(corpus, queries)

            # Evaluate the results
            evaluate(retrieval_result, retriever, qrels, result_file)

    print(f"The code took {time.time() - start_time} seconds to run.")

if __name__ == "__main__":
    main()