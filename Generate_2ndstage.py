import os
import re
import time
import torch
from tqdm import tqdm
from utils import load_multi_queries, save_multi_queries, Q2D_single_passage_generator, CoT_single_passage_generator, CQE_single_passage_generator
from retrying import retry
import openai
import logging
import argparse

def process_subqueries_to_passages(dataset_name, multi_query_version):
    # Define the method and dataset type for file paths
    task_type = dataset_name
    
    # Define input file path (multiQ) and output file path (multiP)
    queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/LC-{task_type}.json"
    passages_file = f"/home/intern/Leon_Kuo/Paper/Generation/multiQ2multiP({multi_query_version})-{task_type}.json"
    
    # Check if the queries file exists
    if os.path.exists(queries_file):
        multi_queries_dict = load_multi_queries(queries_file)
    else:
        print(f"No queries file found at {queries_file}. Exiting.")
        return

    # Load existing passages, or initialize as an empty dictionary
    if os.path.exists(passages_file):
        passages_dict = load_multi_queries(passages_file)
    else:
        passages_dict = {}

    # Generate passages only for sub-queries that haven't been processed
    for key, data in tqdm(multi_queries_dict.items(), desc=f"Processing {dataset_name}", unit="query"):
        if key not in passages_dict or not passages_dict[key].get("expanded"):
            passages_dict[key] = {
                "original": data["original"],
                "expanded": []
            }
            for sub_query in data["expanded"]:
                # Generate a passage for each sub-query
                passage_result = generate_single_passage(sub_query)
                passages_dict[key]["expanded"].append(passage_result["expanded"])

            # Save the updated passages after processing each query
            save_multi_queries(passages_dict, passages_file)

def main(multi_query_version):
    
    
    # Record the start time
    start_time = time.time()
    
    # Set the GPU device
    torch.cuda.set_device(2)
    
    # List of datasets to process
    datasets = [
        'trec-covid',
        'fiqa',
        'dbpedia-entity',
        'nfcorpus',
        'webis-touche2020'
    ]
    
    # Process each dataset
    for dataset in datasets:
        process_subqueries_to_passages(dataset, multi_query_version)
    
    print(f"The code took {time.time() - start_time} seconds to run.")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Process subqueries to generate passages using MMLF')
    parser.add_argument('--multi_query_version', default='var5', help='Version of multi-query to use (e.g., var4, var5)')
    args = parser.parse_args()
    
    main(args.multi_query_version)