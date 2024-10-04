import os
import torch
from tqdm import tqdm
from utils import load_multi_queries, save_multi_queries, load_data
from Generator import (
    langchain_multi_queries_generator, 
    CoT_single_passage_generator, 
    Q2D_single_passage_generator, 
    CQE_single_passage_generator,  
    MILL_multiQ_multiP_generator, 
    MCQE_multiQ_multiP_generator
)
import logging
import argparse


def process_dataset(task, generation_type, generation_stage, queries_file, passages_file=None):
    try:
        # Load multi-query and passage data if they exist
        multi_queries_dict = load_multi_queries(queries_file) if os.path.exists(queries_file) else {}
        passages_dict = load_multi_queries(passages_file) if passages_file and os.path.exists(passages_file) else {}

        # Process data based on generation stage
        if generation_stage == "first":
            process_first_stage(task, generation_type, multi_queries_dict, queries_file)
        elif generation_stage == "second":
            process_second_stage(multi_queries_dict, passages_dict, generation_type, task, passages_file)
        elif generation_stage == "combined":
            process_combined_stage(task, generation_type, multi_queries_dict, passages_dict, queries_file, passages_file)

    except Exception as e:
        logging.error(f"Error processing dataset: {e}")

def process_first_stage(task, generation_type, multi_queries_dict, queries_file):
    """Process the dataset in the first stage of generation."""
    corpus, queries, qrels = load_data(task)
    for key, data in tqdm(queries.items(), desc=f"Processing {task}", unit="data"):
        if key not in multi_queries_dict:
            multi_queries_dict[key] = generate_passage(generation_type, "", data)
            save_multi_queries(multi_queries_dict, queries_file)

def process_second_stage(multi_queries_dict, passages_dict, generation_type, task, passages_file):
    """Process the dataset in the second stage of generation."""
    for key, data in tqdm(multi_queries_dict.items(), desc=f"Processing {task}", unit="data"):
        if key not in passages_dict or not passages_dict[key].get("expanded"):
            passages_dict[key] = {"original": data["original"], "expanded": []}
            for sub_query in data["expanded"]:
                passage_result = generate_passage(generation_type, data["original"], sub_query)
                passages_dict[key]["expanded"].append(passage_result["expanded"])
            save_multi_queries(passages_dict, passages_file)

def process_combined_stage(task, generation_type, multi_queries_dict, passages_dict, queries_file, passages_file):
    """Process the dataset in the combined generation stage."""
    corpus, queries, qrels = load_data(task)
    for key, data in tqdm(queries.items(), desc=f"Processing {task}", unit="data"):
        if key not in multi_queries_dict or key not in passages_dict:
            result = generate_passage(generation_type, "", data)
            multi_queries_dict[key] = {"original": result["original"], "expanded": result["expanded_sub_queries"]}
            passages_dict[key] = {"original": result["original"], "expanded": result["expanded_passages"]}
        save_multi_queries(multi_queries_dict, queries_file)
        save_multi_queries(passages_dict, passages_file)

def generate_passage(generation_type, original_query, query):
    """Generate passage based on generation type."""
    if generation_type == "Q2D":
        return Q2D_single_passage_generator(query)
    elif generation_type == "CoT":
        return CoT_single_passage_generator(query)
    elif generation_type == "CQE":
        return CQE_single_passage_generator(original_query, query)
    elif generation_type == "LC":
        return langchain_multi_queries_generator(query)
    elif generation_type == "QQD(MILL)":
        return MILL_multiQ_multiP_generator(query)
    elif generation_type == "multiCQE":
        return MCQE_multiQ_multiP_generator(query)


def main(generation_type, generation_stage, task, queries_file, passages_file):
    # Set the GPU device
    torch.cuda.set_device(0)

    process_dataset(task, generation_type, generation_stage, queries_file, passages_file)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process queries and passages using MMLF')
    parser.add_argument('--generation_stage', help='first, second, or combined', required=True)
    parser.add_argument('--generation_type', help='CoT, Q2D, LC, multiCQE, QQD(MILL)', required=True)
    parser.add_argument('--task', default='fiqa', help='trec-covid, fiqa, dbpedia-entity, nfcorpus, webis-touche2020')
    parser.add_argument('--queries_file', required=True, help='Path to the queries file')
    parser.add_argument('--passages_file', help='Path to the passages file (optional)')

    args = parser.parse_args()
    main(args.generation_type, args.generation_stage, args.task, args.queries_file, args.passages_file)
