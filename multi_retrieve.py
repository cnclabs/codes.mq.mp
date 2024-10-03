import os
import time
import torch
import logging
from utils import *
import logging
import argparse

def main(method, fusion_method, include_original, concat_original, base_model):
    start_time = time.time()
    
    torch.cuda.set_device(0)
    
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
    
    if include_original==True and concat_original==True:
        fusion_method_name = f"{fusion_method}(include)(concat)"
    elif include_original==True and concat_original==False:
        fusion_method_name = f"{fusion_method}(include)"
    elif include_original==False and concat_original==True:
        fusion_method_name = f"{fusion_method}(concat)"
    else:
        fusion_method_name = f"{fusion_method}"
    
    
    # Iterate over each dataset
    for dataset_name in datasets:
        queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/{method}-{dataset_name}.json"
        
        if include_original == True:
            result_file = f"/home/intern/Leon_Kuo/Paper/Evaluation_Results/result-{method}-{dataset_name}-{fusion_method_name}_withO.json"
        else:
            result_file = f"/home/intern/Leon_Kuo/Paper/Evaluation_Results/result-{method}-{dataset_name}-{fusion_method}_withoutO.json"
        
        # Check if result file exists and is non-empty
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            logging.info(f"Skipping dataset: {dataset_name}. Results already exist and file is non-empty.")
        else:
            logging.info(f"Processing dataset: {dataset_name}")

            # Load data
            corpus, queries, qrels = load_data(dataset_name)

            # Retrieve and combine results
            retrieval_result, retriever = retrieve_and_combine(corpus, queries_file, fusion_method, include_original, concat_original, base_model)

            # Evaluate the results
            evaluate(retrieval_result, retriever, qrels, result_file)

    print(f"The code took {time.time() - start_time} seconds to run.")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Process subqueries to generate passages using MMLF')
    parser.add_argument('--method', default='multiCQE(mP)', help='multiCQE(mP) or multiQ2multiP(var4)')
    parser.add_argument('--fusion_method', default='RRF', help='RRF or ')
    
    parser.add_argument('--include_original', action='store_true', help='..')
    parser.add_argument('--concat_original', action='store_false', help='..')
    
    parser.add_argument('--base_model', default='e5-small-v2', help='e5-small-v2 or contriever')
    
    args = parser.parse_args()

    main(args.method, args.fusion_method, args.include_original, args.concat_original)