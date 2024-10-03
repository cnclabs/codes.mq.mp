import re
import os
import json
import openai
import logging
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from retrying import retry
from transformers import logging as hf_logging
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = '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'

# Set the Transformers log level to error to suppress warnings
hf_logging.set_verbosity_error()


"""Load and Save"""
def load_multi_queries(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def load_data(task):
    out_dir = "/home/intern/Leon_Kuo/QueryExpansion/BEIR/datasets"
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


"""Generate"""
@retry(stop_max_attempt_number=50, wait_fixed=2000)
def generate_multi_queries(query):
    response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[
            {
            "role": "user",
            "content": f"""Generate exactly three different versions of the given user question to retrieve relevant documents from a vector database. The goal is to provide varied perspectives to enhance the effectiveness of similarity search.

Original question: {query}

Format your response in plain text as:

Sub-query 1:

Sub-query 2:

Sub-query 3:"""
            }
        ],
    )
    
    # Extract the content from the response
    reply = response.choices[0].message['content']

    # Regex to capture each sub_query
    pattern_q = r"Sub-query \d+:\s*([\s\S]*?)(?=Sub-query \d+:|$)"

    # Find all sub_queries
    sub_queries = [sq.strip() for sq in re.findall(pattern_q, reply, re.DOTALL)]

    # Clean sub-queries by removing any text after \n\n
    sub_queries = [re.split(r'\n\n', sq)[0].strip() for sq in sub_queries]
    
    # Check if exactly 3 sub-queries are extracted and none are empty
    if len(sub_queries) == 3 and all(sq for sq in sub_queries):
        return {
            "original": query,
            "expanded": sub_queries
        }

    # If the conditions are not met, raise an exception to trigger retry
    print("Failed to generate 3 valid sub-queries. Retrying...")
    raise ValueError("Failed to generate 3 valid sub-queries.")

@retry(stop_max_attempt_number=50, wait_fixed=2000)
def generate_multi_passages(query):
    response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[
            {
            "role": "user",
            "content": f"""Write exactly three different passages to answer the given user question to retrieve relevant documents from a vector database. The goal is to provide varied perspectives to enhance the effectiveness of similarity search.

Original question: {query}

Format your response in plain text as:

Passage 1:

Passage 2:

Passage 3:"""
            }
        ],
    )
    
    # Extract the content from the response
    reply = response.choices[0].message['content']

    # Regex to capture each sub_query
    pattern_q = r"Passage \d+:\s*([\s\S]*?)(?=Passage \d+:|$)"

    # Find all sub_queries
    sub_queries = [sq.strip() for sq in re.findall(pattern_q, reply, re.DOTALL)]

    # Check if exactly 3 sub-queries are extracted and none are empty
    if len(sub_queries) == 3 and all(sq for sq in sub_queries):
        return {
            "original": query,
            "expanded": sub_queries
        }

    # If the conditions are not met, raise an exception to trigger retry
    print("Failed to generate 3 valid sub-queries. Retrying...")
    raise ValueError("Failed to generate 3 valid sub-queries.")

@retry(stop_max_attempt_number=50, wait_fixed=2000)
def generate_queries_and_passages(query):
    response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[
            {
            "role": "user",
            "content": f"""Generate exactly three different versions of the given user question to retrieve relevant documents from a vector database. For each sub-query, also write a passage that answers it. The goal is to provide varied perspectives to enhance the effectiveness of similarity search.

Original question: {query}

Format your response in plain text as:

Sub-query 1:
Passage 1:

Sub-query 2:
Passage 2:

Sub-query 3:
Passage 3:"""
            }
        ],
    )

    # Extract the content from the response
    reply = response.choices[0].message['content']
    
    # Regex to capture each sub-query
    pattern_q = r"Sub-query \d+:\s*([\s\S]*?)(?=Passage \d+:|$)"
    
    # Regex to capture each passage
    pattern_p = r"Passage \d+:\s*([\s\S]*?)(?=Sub-query \d+:|$)"

    # Find all sub_queries and clean them by removing any text after \n\n
    sub_queries = [sq.strip() for sq in re.findall(pattern_q, reply, re.DOTALL)]
    sub_queries = [re.split(r'\n\n', sq)[0].strip() for sq in sub_queries]
    
    # Find all passages
    passages = [ps.strip() for ps in re.findall(pattern_p, reply, re.DOTALL)]
    
    # Check if exactly 3 sub-queries and 3 passages are extracted and none are empty
    if len(sub_queries) == 3 and all(sq for sq in sub_queries) and len(passages) == 3 and all(ps for ps in passages):
        return {
            "original": query,
            "expanded_sub_queries": sub_queries,
            "expanded_passages": passages
        }

    # If the conditions are not met, raise an exception to trigger retry
    print("Failed to generate 3 valid sub-queries and 3 passages. Retrying...")
    raise ValueError("Failed to generate 3 valid sub-queries and 3 passages.")

@retry(stop_max_attempt_number=50, wait_fixed=2000)
def generate_multiCQE(query):
    response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[
            {
            "role": "user",
            "content": f"""You are an AI language model assistant. Your task is to generate exactly three different versions of the given user question (sub-queries) and then write a passage for each sub-query to retrieve relevant documents from a vector database. Each passage should address both the original query and its corresponding sub-query. By generating multiple passages from different perspectives, your goal is to help the user overcome some of the limitations of distance-based similarity search.

Original question: {query}

Format your response in plain text as:

Sub-query 1:
Passage 1:

Sub-query 2:
Passage 2:

Sub-query 3:
Passage 3:"""
            }
        ],
    )

    # Extract the content from the response
    reply = response.choices[0].message['content']
    
    # Regex to capture each sub-query
    pattern_q = r"Sub-query \d+:\s*([\s\S]*?)(?=Passage \d+:|$)"
    
    # Regex to capture each passage
    pattern_p = r"Passage \d+:\s*([\s\S]*?)(?=Sub-query \d+:|$)"

    # Find all sub_queries and clean them by removing any text after \n\n
    sub_queries = [sq.strip() for sq in re.findall(pattern_q, reply, re.DOTALL)]
    sub_queries = [re.split(r'\n\n', sq)[0].strip() for sq in sub_queries]
    
    # Find all passages
    passages = [ps.strip() for ps in re.findall(pattern_p, reply, re.DOTALL)]
    
    # If more than 3 sub-queries and passages are generated, slice to keep only the first 3
    sub_queries = sub_queries[:3]
    passages = passages[:3]
    
    # Check if exactly 3 sub-queries and 3 passages are extracted and none are empty
    if len(sub_queries) == 3 and all(sq for sq in sub_queries) and len(passages) == 3 and all(ps for ps in passages):
        return {
            "original": query,
            "expanded_sub_queries": sub_queries,
            "expanded_passages": passages
        }

    # If the conditions are not met, raise an exception to trigger retry
    print("Failed to generate 3 valid sub-queries and 3 passages. Retrying...")
    raise ValueError("Failed to generate 3 valid sub-queries and 3 passages.")

@retry(stop_max_attempt_number=50, wait_fixed=2000)
def generate_QQD_MILL(query):
    response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[
            {
            "role": "user",
            "content": f"""What sub-queries should be searched to answer the following query? Please generate the sub-queries and write passages to answer these generated queries. 

Query: {query}

Format your response in plain text as:

Sub-query 1:
Passage 1:

Sub-query 2:
Passage 2:

Sub-query 3:
Passage 3:"""
            }
        ],
    )

    # Extract the content from the response
    reply = response.choices[0].message['content']
    
    # Regex to capture each sub-query
    pattern_q = r"Sub-query \d+:\s*([\s\S]*?)(?=Passage \d+:|$)"
    
    # Regex to capture each passage
    pattern_p = r"Passage \d+:\s*([\s\S]*?)(?=Sub-query \d+:|$)"

    # Find all sub_queries and clean them by removing any text after \n\n
    sub_queries = [sq.strip() for sq in re.findall(pattern_q, reply, re.DOTALL)]
    sub_queries = [re.split(r'\n\n', sq)[0].strip() for sq in sub_queries]
    
    # Find all passages
    passages = [ps.strip() for ps in re.findall(pattern_p, reply, re.DOTALL)]
    
    # If more than 3 sub-queries and passages are generated, slice to keep only the first 3
    sub_queries = sub_queries[:3]
    passages = passages[:3]
    
    # Check if exactly 3 sub-queries and 3 passages are extracted and none are empty
    if len(sub_queries) == 3 and all(sq for sq in sub_queries) and len(passages) == 3 and all(ps for ps in passages):
        return {
            "original": query,
            "expanded_sub_queries": sub_queries,
            "expanded_passages": passages
        }

    # If the conditions are not met, raise an exception to trigger retry
    print("Failed to generate 3 valid sub-queries and 3 passages. Retrying...")
    raise ValueError("Failed to generate 3 valid sub-queries and 3 passages.")



"""Fuse"""
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
    """Perform CombSUM by aggregating scores from different queries."""
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
def retrieve_single_query(corpus, json_file, include_original=True, base_model = 'e5-small-v2'):
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


    ############
    # Initialize the retriever here
    if base_model == 'e5-small-v2':
        model = DRES(models.SentenceBERT("intfloat/e5-small-v2"), batch_size=64)
        retriever = EvaluateRetrieval(model, score_function="cos_sim")
    
    if base_model == 'contriever':
    # Initialize the retriever here
        model = DRES(models.SentenceBERT("facebook/contriever"), batch_size=64)
        retriever = EvaluateRetrieval(model, score_function="dot")
    ############


    # Retrieving results for all queries at once using the retriever
    all_results = retriever.retrieve(corpus, all_queries)

    # Extract and organize results
    combined_results = {query_id: all_results[f"{query_id}_single"] for query_id in data.keys()}

    return combined_results, retriever

def retrieve_and_combine(corpus, json_file, fusion_method='RRF', include_original=True, concat_original=False, base_model = 'e5-small-v2'):
    data = load_multi_queries(json_file)
    
    # Build expanded queries with or without the original query
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

    all_queries = {}

    # Prepare queries for batch retrieval
    for query_id, queries in expanded_queries.items():
        for idx, query in enumerate(queries):
            all_queries[f"{query_id}_v{idx}"] = {"text": query}


    ############
    # Initialize the retriever here
    if base_model == 'e5-small-v2':
        model = DRES(models.SentenceBERT("intfloat/e5-small-v2"), batch_size=64)
        retriever = EvaluateRetrieval(model, score_function="cos_sim")
    
    if base_model == 'contriever':
        model = DRES(models.SentenceBERT("facebook/contriever"), batch_size=64)
        retriever = EvaluateRetrieval(model, score_function="dot")
    ############


    # Retrieving results for all queries at once using the retriever
    all_results = retriever.retrieve(corpus, all_queries)

    # Applying the selected fusion method to combine results
    combined_results = {}
    for query_id, queries in expanded_queries.items():
        search_results = {f"{query_id}_v{idx}": all_results[f"{query_id}_v{idx}"] for idx in range(len(queries)) if f"{query_id}_v{idx}" in all_results}
        
        if fusion_method == 'RRF':
            combined_results[query_id] = reciprocal_rank_fusion(search_results)
        elif fusion_method == 'CombSUM':
            combined_results[query_id] = CombSUM(search_results)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    return combined_results, retriever


"""Evaluate"""
def evaluate(results, retriever, qrels, output_file):
    logging.info("Evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, retriever.k_values)
    all_metrics = {**ndcg, **_map, **recall, **precision}
    print(all_metrics)
    
    # Save the evaluation metrics to a file
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    return all_metrics


if __name__ == "__main__":
    print(generate_multi_queries("Effiel"))