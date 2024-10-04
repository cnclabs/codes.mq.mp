import os
import time
import torch
from tqdm import tqdm
from utils import load_multi_queries, save_multi_queries, load_data
from Generator import langchain_multi_queries_generator, CoT_single_passage_generator, Q2D_single_passage_generator, CQE_single_passage_generator,  MILL_multiQ_multiP_generator, MCQE_multiQ_multiP_generator
from retrying import retry
import openai
import logging
import argparse

def process_dataset(task, generation_type, generation_stage):
    try:
        # Determine the file paths based on the generation stage and type
        if generation_stage == "first":
            queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/{generation_type}-{task}.json"
            passages_file = None
        elif generation_stage == "second":
            queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/LC-{task}.json"
            passages_file = f"/home/intern/Leon_Kuo/Paper/Generation/multiQ2multiP({generation_type})-{task}.json"
        else:
            queries_file = f"/home/intern/Leon_Kuo/Paper/Generation/{generation_type}(mQ)-{task}.json"
            passages_file = f"/home/intern/Leon_Kuo/Paper/Generation/{generation_type}(mP)-{task}.json"

        # Load queries
        multi_queries_dict = load_multi_queries(queries_file) if os.path.exists(queries_file) else {}

        # Load passages for second or combined stage
        passages_dict = load_multi_queries(passages_file) if os.path.exists(passages_file) else {}

        if generation_stage == "second":
            # Process dataset
            for key, data in tqdm(multi_queries_dict.items(), desc=f"Processing {task}", unit="data"):  #??
                if key not in passages_dict or not passages_dict[key].get("expanded"):
                    passages_dict[key] = {"original": data["original"], "expanded": []}
                    for sub_query in data["expanded"]:
                        if generation_type == "Q2D": 
                            passage_result = Q2D_single_passage_generator(sub_query)
                        elif generation_type == "CoT": 
                            passage_result = CoT_single_passage_generator(sub_query)
                        elif generation_type == "CQE": 
                            passage_result = CQE_single_passage_generator(sub_query)
                        passages_dict[key]["expanded"].append(passage_result["expanded"])

                    save_multi_queries(passages_dict, passages_file)
        elif generation_stage == "first":
            corpus, queries, qrels = load_data(task_type)
            # Process dataset
            for key, data in tqdm(queries.items(), desc=f"Processing {task}", unit="data"):  #??
                if key not in multi_queries_dict:
                    # Generation based on type
                    if generation_type == "CoT":
                        multi_queries_dict[key] = CoT_single_passage_generator(data)
                    elif generation_type == "Q2D":
                        multi_queries_dict[key] = Q2D_single_passage_generator(data)
                    elif generation_type == "LC":
                        multi_queries_dict[key] = langchain_multi_queries_generator(data)

                    # Save after processing all queries
                    save_multi_queries(multi_queries_dict, queries_file) #X
        elif generation_stage == "combined":
            corpus, queries, qrels = load_data(task_type)
            # Process dataset
            for key, data in tqdm(queries.items(), desc=f"Processing {task}", unit="data"):  #??
                if key not in multi_queries_dict or key not in passages_dict:
                    if generation_type == "QQD(MILL)": 
                        result =  MILL_multiQ_multiP_generator(data)
                    elif generation_type == "multiCQE": 
                        result = MCQE_multiQ_multiP_generator(data)
                    multi_queries_dict[key] = {"original": result["original"], "expanded": result["expanded_sub_queries"]}
                    passages_dict[key] = {"original": result["original"], "expanded": result["expanded_passages"]}
                
                    # Save after processing all queries
                    save_multi_queries(multi_queries_dict, queries_file) 
                    save_multi_queries(passages_dict, passages_file)
    except Exception as e:
        logging.error(f"Error processing dataset: {e}")

def main(generation_type, generation_stage, task):
    # Set the GPU device
    torch.cuda.set_device(0)

    process_dataset(task, generation_type, generation_stage)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process queries and passages using MMLF')
    parser.add_argument('--generation_stage', help='first, second, or combined')
    parser.add_argument('--generation_type', help='CoT, Q2D, LC, multiCQE, QQD(MILL)')
    parser.add_argument('--task', default='fiqa', help='trec-covid, fiqa, dbpedia-entity, nfcorpus, webis-touche2020')

    args = parser.parse_args()

    main(args.generation_type, args.generation_stage, args.task)
