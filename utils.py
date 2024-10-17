import re
import os
import json
import logging
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from transformers import logging as hf_logging
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

"""Load and Save"""
def load_multi_queries(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def load_data(task):
    out_dir = "/path/to/datasets"
    data_path = os.path.join(out_dir, task)
    if not os.path.exists(data_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{task}.zip"
        util.download_and_unzip(url, out_dir)
        print(f"Dataset downloaded and extracted to: {data_path}")
    else:
        print(f"Dataset already exists at: {data_path}")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    return corpus, queries, qrels

def save_multi_queries(multi_queries, filename):
    with open(filename, 'w') as f:
        json.dump(multi_queries, f, indent=4)

"""Fuse"""
# The following reciprocal_rank_fusion function is adapted from https://github.com/Raudaschl/rag-fusion.
def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            fused_scores[doc] += 1 / (rank + k)
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

def CombSUM(search_results_dict):
    aggregated_scores = {}
    
    # Iterate over queries and document scores
    for query, doc_scores in search_results_dict.items():
        for doc, score in doc_scores.items():
            if doc not in aggregated_scores:
                aggregated_scores[doc] = 0  # Initialize score for new documents
            aggregated_scores[doc] += score  # Sum the scores across queries
    
    # Sort documents by aggregated score in descending order (higher scores first)
    reranked_results = {doc: score for doc, score in sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)}
    
    return reranked_results

"""Retrieve"""
# Helper function to initialize the retriever
def initialize_retriever(base_model: str) -> EvaluateRetrieval:
    if base_model == 'e5-small-v2':
        model = DRES(models.SentenceBERT("intfloat/e5-small-v2"), batch_size=64)
        return EvaluateRetrieval(model, score_function="cos_sim")
    elif base_model == 'contriever':
        model = DRES(models.SentenceBERT("facebook/contriever"), batch_size=64)
        return EvaluateRetrieval(model, score_function="dot")
    else:
        raise ValueError(f"Unsupported base model: {base_model}")

# Function to prepare queries for retrieval
def prepare_queries(data: dict, include_original: bool, concat_original: bool) -> dict:
    expanded_queries = {}
    for key, value in data.items():
        # Handle cases where 'expanded' is a string instead of a list
        if isinstance(value['expanded'], str):
            expanded_list = [value['expanded']]
        else:
            expanded_list = value['expanded']

        
        if include_original and concat_original:
            expanded_queries[key] = [value['original']] + [f"{value['original']} [SEP] {expanded}" for expanded in expanded_list]
        elif include_original and not concat_original:
            expanded_queries[key] = [value['original']] + expanded_list
        elif not include_original and concat_original:
            expanded_queries[key] = [f"{value['original']} [SEP] {expanded}" for expanded in expanded_list]
        else:
            expanded_queries[key] = expanded_list

    return expanded_queries

# Single query retrieval function
def retrieve_single_query(corpus: str, json_file: str, include_original: bool, base_model: str):
    data = load_multi_queries(json_file)
    
    all_queries = {}

    for query_id, value in data.items():
        original_query = value['original']
        expanded_queries = value['expanded']
        
        # Ensure expanded_queries is a list for consistency
        if isinstance(expanded_queries, str):
            expanded_queries = [expanded_queries]
        
        # Case [1]: Only one expanded query
        if len(expanded_queries) == 1:
            if include_original:
                # Concatenate original query with the expanded query
                text_for_retrieval = f"{original_query} [SEP] {expanded_queries[0]}"
            else:
                # Only use the expanded query
                text_for_retrieval = expanded_queries[0]
        # Case [2]: Multiple expanded queries
        else:
            if include_original:
                # Concatenate original query with all expanded queries
                text_for_retrieval = f"{original_query} [SEP] " + " [SEP] ".join(expanded_queries)
            else:
                # Only use all expanded queries
                text_for_retrieval = " [SEP] ".join(expanded_queries)

        # Prepare a unique identifier for each query
        query_key = f"{query_id}_single"
        all_queries[query_key] = {"text": text_for_retrieval}

    # Initialize retriever
    retriever = initialize_retriever(base_model)

    # Retrieve results
    all_results = retriever.retrieve(corpus, all_queries)

    # Extract and organize results
    combined_results = {query_id: all_results[f"{query_id}_single"] for query_id in data.keys()}

    return combined_results, retriever

# Multiple query retrieval and fusion function
def retrieve_and_combine(corpus: str, json_file: str, fusion_method: str, include_original: bool, concat_original: bool, base_model: str):
    data = load_multi_queries(json_file)

    # Prepare queries
    expanded_queries = prepare_queries(data, include_original, concat_original)

    all_queries = {f"{query_id}_v{idx}": {"text": query} 
                   for query_id, queries in expanded_queries.items() 
                   for idx, query in enumerate(queries)}

    # Initialize retriever
    retriever = initialize_retriever(base_model)

    # Retrieve results for all queries
    all_results = retriever.retrieve(corpus, all_queries)

    # Apply the selected fusion method to combine results
    combined_results = {}
    for query_id, queries in expanded_queries.items():
        search_results = {f"{query_id}_v{idx}": all_results.get(f"{query_id}_v{idx}") 
                          for idx in range(len(queries))}

        if fusion_method == 'RRF':
            combined_results[query_id] = reciprocal_rank_fusion(search_results)
        elif fusion_method == 'CombSUM':
            combined_results[query_id] = CombSUM(search_results)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    return combined_results, retriever

# General retrieval function (for use when no query modification is needed)
def pure_retrieve(corpus: str, json_file: str, base_model: str):
    # Initialize retriever
    retriever = initialize_retriever(base_model)

    # Retrieve results
    retrieval_result = retriever.retrieve(corpus, json_file)
    
    return retrieval_result, retriever

"""Evaluate"""
def evaluate(results, retriever, qrels, output_file):
    logging.info("Evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, retriever.k_values)
    ll_metrics = {**ndcg, **_mp, **recll, **precision}
    print(ll_metrics)
    
    # Save the evaluation metrics to a file
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    return all_metrics
