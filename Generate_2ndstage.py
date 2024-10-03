import os
import re
import time
import torch
from tqdm import tqdm
from utils import load_multi_queries, save_multi_queries
from Generator import Q2D_single_passage_generator, CoT_single_passage_generator, CQE_single_passage_generator
from retrying import retry
import openai
import logging
import argparse

def process_subqueries_to_passages(task, second_stage_generation):
    # Define input file path (multiQ) and output file path (multiP)
    queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/LC-{task}.json"
    passages_file = f"/home/intern/Leon_Kuo/Paper/Generation/multiQ2multiP({second_stage_generation})-{task}.json"
    
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
    for key, data in tqdm(multi_queries_dict.items(), desc=f"Processing {task}", unit="query"):
        if key not in passages_dict or not passages_dict[key].get("expanded"):
            passages_dict[key] = {
                "original": data["original"],
                "expanded": []
            }
            for sub_query in data["expanded"]:
                # Generate a passage for each sub-query
                if second_stage_generation == "Q2D":
                    passage_result = Q2D_single_passage_generator(sub_query)
                if second_stage_generation == "CoT":
                    passage_result = CoT_single_passage_generator(sub_query)
                if second_stage_generation == "CQE":
                    passage_result = CQE_single_passage_generator(sub_query)
                
                passages_dict[key]["expanded"].append(passage_result["expanded"])

            # Save the updated passages after processing each query
            save_multi_queries(passages_dict, passages_file)

def main(second_stage_generation, task):   
    # Record the start time
    start_time = time.time()
    
    # Set the GPU device
    torch.cuda.set_device(2)
    
    process_subqueries_to_passages(task, second_stage_generation)
    
    print(f"The code took {time.time() - start_time} seconds to run.")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Process subqueries to generate passages using MMLF')
    parser.add_argument('--task', default='fiqa', help='trec-covid, fiqa, dbpedia-entity, nfcorpus, webis-touche2020')
    parser.add_argument('--second_stage_generation', default='Q2D', help='Version of multi-query to use (e.g., Q2D, CoT, CQE)')
    
    args = parser.parse_args()
    
    main(args.second_stage_generation, args.task)