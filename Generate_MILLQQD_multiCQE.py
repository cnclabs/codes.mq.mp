import os
import time
import torch
from tqdm import tqdm
from utils import load_multi_queries, save_multi_queries, load_data, generate_QQD_MILL, generate_multiCQE
import logging
import argparse

def process_dataset(dataset_name, method):
    # Define the method and dataset type for file paths
    task_type = dataset_name
    
    # Define file paths for sub-queries and passages
    sub_queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/{method}(mQ)-{task_type}.json"
    passages_file = f"/home/intern/Leon_Kuo/Paper/Generation/{method}(mP)-{task_type}.json"
    
    # Load existing sub-queries and passages, or initialize as empty dictionaries
    if os.path.exists(sub_queries_file):
        multi_queries_dict = load_multi_queries(sub_queries_file)
    else:
        multi_queries_dict = {}
    
    if os.path.exists(passages_file):
        passages_dict = load_multi_queries(passages_file)
    else:
        passages_dict = {}

    # Load the corpus, queries, and qrels
    corpus, queries, qrels = load_data(task_type)
    
    # Generate only for queries that don't have their versions or have an error
    for key, query in tqdm(queries.items(), desc=f"Processing {dataset_name}", unit="query"):
        if key not in multi_queries_dict or key not in passages_dict:
            # Generate sub-queries and passages
            if method == "MILLQQD":
                result = generate_QQD_MILL(query)
            if method == "multiCQE":
                result = generate_multiCQE(query)
            
            # Store the results in the respective dictionaries
            multi_queries_dict[key] = {
                "original": result["original"],
                "expanded": result["expanded_sub_queries"]
            }
            passages_dict[key] = {
                "original": result["original"],
                "expanded": result["expanded_passages"]
            }
            
            # Save the updated results after processing each query
            save_multi_queries(multi_queries_dict, sub_queries_file)
            save_multi_queries(passages_dict, passages_file)

def main(method):
    # Record the start time
    start_time = time.time()
    
    # Set the GPU device
    torch.cuda.set_device(2)
    
    # List of datasets to process
    datasets = [
        'scidocs'
    ]
    
    """ 'scifact',  #
        'nfcorpus', #
        'fiqa',         ##
        'trec-covid',       ###
        'dbpedia-entity',   ###
        'webis-touche2020'  ### """
    
    # Process each dataset
    for dataset in datasets:
        process_dataset(dataset, method)
    
    print(f"The code took {time.time() - start_time} seconds to run.")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Process subqueries to generate passages using MMLF')
    parser.add_argument('--method', default='multiCQE', help='multiCQE or QQD(MILL)')
    args = parser.parse_args()
    
    main(args.method)