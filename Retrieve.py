import os
import time
import torch
import logging
from utils import load_data, evaluate, retrieve_and_combine, retrieve_single_query, pure_retrieve
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import argparse

def main(retrieval_type, fusion_method, include_original, concat_original, base_model, task, queries_file, result_file):
    # Check for CUDA availability and set device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    else:
        logging.warning("CUDA is not available. Running on CPU.")

    # Configure logging
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    # Check if result file exists and is non-empty
    if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
        logging.info(f"Skipping dataset: {task}. Results already exist and file is non-empty.")
    else:
        try:
            logging.info(f"Processing dataset: {task}")
            corpus, queries, qrels = load_data(task)
        except Exception as e:
            logging.error(f"Failed to load data for task {task}: {e}")
            return

        # Retrieve results based on retrieval type
        try:
            if retrieval_type == 'multiple':
                retrieval_result, retriever = retrieve_and_combine(corpus, queries_file, fusion_method, include_original, concat_original, base_model)
            elif retrieval_type == 'single':
                retrieval_result, retriever = retrieve_single_query(corpus, queries_file, concat_original, base_model)
            elif retrieval_type == 'rawQ':
                retrieval_result, retriever = pure_retrieve(corpus, queries, base_model)
        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            return

        # Evaluate the results
        try:
            evaluate(retrieval_result, retriever, qrels, result_file)
            logging.info(f"Completed processing of {task}. Results saved to {result_file}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process subqueries to generate passages using MMLF')
    
    # Define arguments for both methods
    #parser.add_argument('--method', default=None, help='multiCQE(mP), multiQ2multiP(var4), QQD(MILL)(mP), or singleP')
    parser.add_argument('--retrieval_type', help='single, multiple, rawQ')
    parser.add_argument('--fusion_method', default=None, help='RRF, fusion method, or None for single query retrieval')
    parser.add_argument('--include_original', action='store_false', help='Include original queries in the retrieval process')
    parser.add_argument('--concat_original', action='store_false', help='Concatenate original queries with expanded queries')
    parser.add_argument('--base_model', default='e5-small-v2', help='e5-small-v2, contriever, or other base models')
    parser.add_argument('--task', help='trec-covid, fiqa, dbpedia-entity, nfcorpus, webis-touche2020')
    
    # New arguments for file paths
    parser.add_argument('--queries_file', default=None, help='Path to the queries file')
    parser.add_argument('--result_file', default=None, help='Path to the result file')

    args = parser.parse_args()

    # Call main function with parsed arguments
    #main(args.method, args.retrieval_type, args.fusion_method, args.include_original, args.concat_original, args.base_model, args.task, args.queries_file, args.result_file)
    main(args.retrieval_type, args.fusion_method, args.include_original, args.concat_original, args.base_model, args.task, args.queries_file, args.result_file)
