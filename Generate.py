"""
File: Generate.py

Description:
    This script processes datasets to generate multiple queries and passages using various generation techniques 
    like CoT, Q2D, MQR, and MCQE. It operates in three stages: first, second, and combined. The script facilitates 
    query-passage generation for tasks such as TREC-COVID, FiQA, DBPedia-Entity, and more.

    The generation process is split into:
        - First stage: Generates multi-queries for each original query.
        - Second stage: Generates expanded passages based on sub-queries.
        - Combined stage: Combines query and passage generation in one step.

Usage:
    To run the script, specify the generation stage, generation type, and the task dataset:
    
    Example command:
    ```bash
    python Generate.py --llm_model meta-llama/Meta-Llama-3-8B-Instruct --generation_stage first --generation_type CoT --task trec-covid --queries_file queries.json
    ```

Dependencies:
    - PyTorch
    - TQDM
    - argparse
    - Custom modules from `utils.py` and `Generator.py` (ensure these are in the same directory or importable)
    
Acknowledgments:
    - The query and passage generation methods are based on MMLF and customized generation techniques.
    
License:
    MIT License. See LICENSE file for more details.

"""

import os
import torch
from tqdm import tqdm
from utils import load_multi_queries, save_multi_queries, load_data
from Generator import (
    MQR_multi_queries_generator, 
    CoT_single_passage_generator, 
    Q2D_single_passage_generator, 
    CQE_single_passage_generator,  
    QQD_multi_passage_generator, 
    MCQE_multi_passage_generator
)
import logging
import argparse


def process_dataset(task, generation_type, generation_stage, queries_file, passages_file, llm):
    try:
        # Load multi-query and passage data if they exist
        multi_queries_dict = load_multi_queries(queries_file) if os.path.exists(queries_file) else {}
        passages_dict = load_multi_queries(passages_file) if passages_file and os.path.exists(passages_file) else {}

        # Process data based on generation stage
        if generation_stage == "first":
            process_first_stage(task, generation_type, multi_queries_dict, queries_file, llm)
        elif generation_stage == "second":
            process_second_stage(multi_queries_dict, passages_dict, generation_type, task, passages_file, llm)
        elif generation_stage == "combined":
            process_combined_stage(task, generation_type, multi_queries_dict, passages_dict, queries_file, passages_file, llm)

    except Exception as e:
        logging.error(f"Error processing dataset: {e}")

def process_first_stage(task, generation_type, multi_queries_dict, queries_file, llm):
    """Process the dataset in the first stage of generation."""
    corpus, queries, qrels = load_data(task)
    # Generate only for queries that don't have their versions or have an error
    for key, data in tqdm(queries.items(), desc=f"Processing {task}", unit="data"):
        if key not in multi_queries_dict:
            multi_queries_dict[key] = generate_passage(generation_type, "", data, llm)
            save_multi_queries(multi_queries_dict, queries_file)

def process_second_stage(multi_queries_dict, passages_dict, generation_type, task, passages_file, llm):
    """Process the dataset in the second stage of generation."""
    for key, data in tqdm(multi_queries_dict.items(), desc=f"Processing {task}", unit="data"):
        if key not in passages_dict or not passages_dict[key].get("expanded"):
            passages_dict[key] = {"original": data["original"], "expanded": []}
            for sub_query in data["expanded"]:
                passage_result = generate_passage(generation_type, data["original"], sub_query, llm)
                passages_dict[key]["expanded"].append(passage_result["expanded"])
            save_multi_queries(passages_dict, passages_file)

def process_combined_stage(task, generation_type, multi_queries_dict, passages_dict, queries_file, passages_file, llm):
    """Process the dataset in the combined generation stage."""
    corpus, queries, qrels = load_data(task)
    for key, data in tqdm(queries.items(), desc=f"Processing {task}", unit="data"):
        if key not in multi_queries_dict or key not in passages_dict:
            result = generate_passage(generation_type, "", data, llm)
            multi_queries_dict[key] = {"original": result["original"], "expanded": result["expanded_sub_queries"]}
            passages_dict[key] = {"original": result["original"], "expanded": result["expanded_passages"]}
        save_multi_queries(multi_queries_dict, queries_file)
        save_multi_queries(passages_dict, passages_file)

def generate_passage(generation_type, original_query, query, llm):
    """Generate passage based on generation type."""
    if generation_type == "Q2D":
        return Q2D_single_passage_generator(query, llm)
    elif generation_type == "CoT":
        return CoT_single_passage_generator(query, llm)
    elif generation_type == "CQE":
        return CQE_single_passage_generator(original_query, query, llm)
    elif generation_type == "MQR":
        return MQR_multi_queries_generator(query, llm)
    elif generation_type == "QQD":
        return QQD_multi_passage_generator(query, llm)
    elif generation_type == "MCQE":
        return MCQE_multi_passage_generator(query, llm)


def main(generation_type, generation_stage, task, queries_file, passages_file, llm):
    process_dataset(task, generation_type, generation_stage, queries_file, passages_file, llm)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process queries and passages using MMLF')
    parser.add_argument('--llm_model', default=None, help='llm model type', required=True)
    parser.add_argument('--generation_stage', help='first, second, or combined', required=True)
    parser.add_argument('--generation_type', help='CoT, Q2D, MQR, MCQE, QQD', required=True)
    parser.add_argument('--task', help='trec-covid, fiqa, dbpedia-entity, nfcorpus, webis-touche2020')
    parser.add_argument('--queries_file', default=None, help='Path to the queries file')
    parser.add_argument('--passages_file', default=None, help='Path to the passages file (optional)')

    args = parser.parse_args()
    main(args.generation_type, args.generation_stage, args.task, args.queries_file, args.passages_file, args.llm_model)
