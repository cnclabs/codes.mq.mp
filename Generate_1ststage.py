import os
import re
import time
import torch
from tqdm import tqdm
from utils import load_multi_queries, load_data,save_multi_queries
from Generator import langchain_multi_queries_generator, CoT_single_passage_generator, Q2D_single_passage_generator
from retrying import retry
import openai
import logging
import argparse

def process_dataset(task, first_stage_generation):
    # Define file path for passage
    queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/{first_stage_generation}-{task}.json"
    
    # Check if the queries file exists
    if os.path.exists(queries_file):
        multi_queries_dict = load_multi_queries(queries_file)
    else:
        multi_queries_dict = {}

    corpus, queries, qrels = load_data(task)
    
    # Generate only for queries that don't have their versions or have an error
    for key, query in tqdm(queries.items(), desc=f"Processing {task}", unit="query"):
        if key not in multi_queries_dict:
            if first_stage_generation == "CoT":
                multi_queries_dict[key] = CoT_single_passage_generator(query)
            if first_stage_generation == "Q2D":           
                multi_queries_dict[key] = Q2D_single_passage_generator(query)
            if first_stage_generation == "LC":           
                multi_queries_dict[key] = langchain_multi_queries_generator(query)
            
            # Save the updated queries after processing each query
            save_multi_queries(multi_queries_dict, queries_file)

def main(first_stage_generation, task):
    # Record the start time
    start_time = time.time()
    
    # Set the GPU device
    torch.cuda.set_device(0)
    
    process_dataset(task, first_stage_generation)
    
    print(f"The code took {time.time() - start_time} seconds to run.")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Process subqueries to generate passages using MMLF')
    parser.add_argument('--first_stage_generation', default='CoT', help='CoT or LC or Q2D')
    parser.add_argument('--task', default='fiqa', help='trec-covid, fiqa, dbpedia-entity, nfcorpus, webis-touche2020')
    
    args = parser.parse_args()
    
    main(args.first_stage_generation, args.task)