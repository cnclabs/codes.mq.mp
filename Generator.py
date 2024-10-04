from retrying import retry
import openai
import re
from config import OPENAI_KEY

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = OPENAI_KEY

@retry(stop_max_attempt_number=50, wait_fixed=2000)
def Q2D_single_passage_generator(query):
        response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[
            {
            "role": "user",
            "content": f"""Please write a passage to answer the query. 

            Query: {query}

            Format your response in plain text as:

            Passage:"""
            }
        ],
    )
        
        # Extract the content from the response
        reply = response.choices[0].message['content']

        # Regex to capture the passage
        pattern_p = r"Passage:\s*([\s\S]*?)$"

        # Find the passage
        passage = re.findall(pattern_p, reply, re.DOTALL)

        # Check if the passage is not empty
        if passage and passage[0].strip():
            return {
                "original": query,
                "expanded": passage[0].strip()
            }

        # If the passage is empty or not found, raise an exception to trigger retry
        print("Failed to generate a valid passage. Retrying...")
        raise ValueError("Failed to generate a valid passage.")

@retry(stop_max_attempt_number=50, wait_fixed=2000)
def MILL_multiQ_multiP_generator(query):
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
def CQE_single_passage_generator(original_query, sub_query):
        response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[
            {
            "role": "user",
            "content": f"""Please write a passage to answer the following user questions simultaneously.

            Question 1: {original_query}
            Question 2: {sub_query}

            Format your response in plain text as:

            Passage:"""
            }
        ],
    )
        
        # Extract the content from the response
        reply = response.choices[0].message['content']

        # Regex to capture the passage
        pattern_p = r"Passage:\s*([\s\S]*?)$"

        # Find the passage
        passage = re.findall(pattern_p, reply, re.DOTALL)

        # Check if the passage is not empty
        if passage and passage[0].strip():
            return {
                "original": original_query,
                "expanded": passage[0].strip()
            }

        # If the passage is empty or not found, raise an exception to trigger retry
        print("Failed to generate a valid passage. Retrying...")
        raise ValueError("Failed to generate a valid passage.")

@retry(stop_max_attempt_number=50, wait_fixed=2000)
def langchain_multi_queries_generator(query):
    response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[
            {
            "role": "user",
            "content": f"""You are an AI language model assistant. Your task is to generate exactly three different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.

            Original question: {query}

            Format your response in plain text as:

            Sub-query 1:

            Sub-query 2:

            Sub-query 3:
            """
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
def CoT_single_passage_generator(query):
        response = openai.ChatCompletion.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[
            {
            "role": "user",
            "content": f"""Answer the following query:

            Query: {query}

            Provide the rationale before answering, and format your response in plain text as:

            Rationale:
            Answer:"""
            }
        ],
    )
        
        # Extract the content from the response
        reply = response.choices[0].message['content']

        # print(reply)

        # Regex to capture the rationale
        pattern_r = r"Rationale:\s*([\s\S]*?)(?=Answer:|$)"
        
        # Regex to capture the answer
        pattern_a = r"Answer:\s*([\s\S]*?)$"

        # Find the passage
        rationale = re.findall(pattern_r, reply, re.DOTALL)
        answer = re.findall(pattern_a, reply, re.DOTALL)

        # Check if both the rationale and the answer are non-empty
        if rationale and rationale[0].strip() and answer and answer[0].strip():
            return {
                "original": query,
                "expanded": f"{rationale[0].strip()}\n\n{answer[0].strip()}"
            }

        # If the passage is empty or not found, raise an exception to trigger retry
        print("Failed to generate a valid passage. Retrying...")
        raise ValueError("Failed to generate a valid passage.")

@retry(stop_max_attempt_number=50, wait_fixed=2000)
def MCQE_multiQ_multiP_generator(query):
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