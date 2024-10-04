import os
import time
import torch
from tqdm import tqdm
from utils import load_multi_queries, save_multi_queries, load_data
from Generator import langchain_multi_queries_generator, CoT_single_passage_generator, Q2D_single_passage_generator, CQE_single_passage_generator, MILL_generator, MCQE_generator
from retrying import retry
import openai
import logging
import argparse

def process_dataset(task, generation_type, generation_stage):
    # Determine the file paths based on the generation stage and type
    if generation_stage == "first":
        queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/{generation_type}-{task}.json"
        passages_file = None  # First stage doesn't generate passages
    elif generation_stage == "second":
        queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/LC-{task}.json"
        passages_file = f"/home/intern/Leon_Kuo/Paper/Generation/multiQ2multiP({generation_type})-{task}.json"
    else:  # For "combined" stage with subqueries and passages
        queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/{generation_type}(mQ)-{task}.json"
        passages_file = f"/home/intern/Leon_Kuo/Paper/Generation/{generation_type}(mP)-{task}.json"

    # Load queries
    if os.path.exists(queries_file):
        multi_queries_dict = load_multi_queries(queries_file)
    else:
        multi_queries_dict = {}

    # Load passages for second or combined stage
    if passages_file and os.path.exists(passages_file):
        passages_dict = load_multi_queries(passages_file)
    else:
        passages_dict = {} if passages_file else None

    # Load the corpus, queries, and qrels
    corpus, queries, qrels = load_data(task)

    # Process dataset
    for key, query in tqdm(queries.items(), desc=f"Processing {task}", unit="query"):
        if key not in multi_queries_dict or (passages_dict and key not in passages_dict):
            if generation_stage == "first":
                if generation_type == "CoT":
                    multi_queries_dict[key] = CoT_single_passage_generator(query)
                elif generation_type == "Q2D":
                    multi_queries_dict[key] = Q2D_single_passage_generator(query)
                elif generation_type == "LC":
                    multi_queries_dict[key] = langchain_multi_queries_generator(query)
                save_multi_queries(multi_queries_dict, queries_file)

            elif generation_stage == "second":
                passages_dict[key] = {"original": multi_queries_dict[key]["original"], "expanded": []}
                for sub_query in multi_queries_dict[key]["expanded"]:
                    if generation_type == "Q2D":
                        passage_result = Q2D_single_passage_generator(sub_query)
                    elif generation_type == "CoT":
                        passage_result = CoT_single_passage_generator(sub_query)
                    elif generation_type == "CQE":
                        passage_result = CQE_single_passage_generator(sub_query)
                    passages_dict[key]["expanded"].append(passage_result["expanded"])
                save_multi_queries(passages_dict, passages_file)

            elif generation_stage == "combined":
                if generation_type == "MILLQQD":
                    result = MILL_generator(query)
                elif generation_type == "multiCQE":
                    result = MCQE_generator(query)
                multi_queries_dict[key] = {"original": result["original"], "expanded": result["expanded_sub_queries"]}
                passages_dict[key] = {"original": result["original"], "expanded": result["expanded_passages"]}
                save_multi_queries(multi_queries_dict, queries_file)
                save_multi_queries(passages_dict, passages_file)

def main(generation_type, generation_stage, task):
    # Record the start time
    start_time = time.time()

    # Set the GPU device
    torch.cuda.set_device(0 if generation_stage == "first" else 2)

    process_dataset(task, generation_type, generation_stage)

    print(f"The code took {time.time() - start_time} seconds to run.")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process queries and passages using MMLF')
    parser.add_argument('--generation_stage', default='first', help='first, second, or combined')
    parser.add_argument('--generation_type', default='CoT', help='CoT, Q2D, LC, multiCQE, MILLQQD')
    parser.add_argument('--task', default='fiqa', help='trec-covid, fiqa, dbpedia-entity, nfcorpus, webis-touche2020')

    args = parser.parse_args()

    main(args.generation_type, args.generation_stage, args.task)
