import os
import time
import torch
import logging
from utils import load_data, evaluate, retrieve_and_combine, retrieve_single_query, retrieve
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import argparse

def main(method, retrieval_type, fusion_method, include_original, concat_original, base_model, task):
    start_time = time.time()
    
    # Set the GPU device
    torch.cuda.set_device(0)

    # Configure logging
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    # Define result file and queries file based on retrieval type and flags
    if retrieval_type == 'rawQ':
        result_file = f"/home/intern/Leon_Kuo/Paper/Evaluation_Results/result-rawQ-{task}.json"
    else:
        queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/{method}-{task}.json"
        if retrieval_type == 'multiple':
            fusion_method_name = f"{fusion_method}(include)" if include_original else fusion_method
            fusion_method_name += "(concat)" if concat_original else ""
            result_file = f"/home/intern/Leon_Kuo/Paper/Evaluation_Results/result-{method}-{task}-{fusion_method_name}.json"
        elif retrieval_type == 'single':
            concat_suffix = "Concat_withO" if concat_original else "Concat_withoutO"
            result_file = f"/home/intern/Leon_Kuo/Paper/Evaluation_Results/result-{method}-{task}-{concat_suffix}.json"
    
    # Check if result file exists and is non-empty
    if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
        logging.info(f"Skipping dataset: {task}. Results already exist and file is non-empty.")
        return

    logging.info(f"Processing dataset: {task}")

    # Load data
    corpus, queries, qrels = load_data(task)

    # Retrieve results based on retrieval type
    if retrieval_type == 'multiple':
        retrieval_result, retriever = retrieve_and_combine(corpus, queries_file, fusion_method, include_original, concat_original, base_model)
    elif retrieval_type == 'single':
        retrieval_result, retriever = retrieve_single_query(corpus, queries_file, concat_original, base_model)
    elif retrieval_type == 'rawQ':
        retrieval_result, retriever = retrieve(corpus, queries_file, base_model)
    # Evaluate the results
    evaluate(retrieval_result, retriever, qrels, result_file)
    logging.info(f"Completed processing of {task}. Results saved to {result_file}")
    
    print(f"The code took {time.time() - start_time} seconds to run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process subqueries to generate passages using MMLF')
    
    # Define arguments for both methods
    parser.add_argument('--method', default='multiCQE(mP)', help='multiCQE(mP), multiQ2multiP(var4), or singleP')
    parser.add_argument('--retrieval_type', default='single', help='single, multiple, rawQ')
    parser.add_argument('--fusion_method', default=None, help='RRF, fusion method, or None for single query retrieval')
    parser.add_argument('--include_original', action='store_true', help='Include original queries in the retrieval process')
    parser.add_argument('--concat_original', action='store_true', help='Concatenate original queries with expanded queries')
    parser.add_argument('--base_model', default='e5-small-v2', help='e5-small-v2, contriever, or other base models')
    parser.add_argument('--task', default='fiqa', help='trec-covid, fiqa, dbpedia-entity, nfcorpus, webis-touche2020')
    
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args.method, args.retrieval_type, args.fusion_method, args.include_original, args.concat_original, args.base_model, args.task)
