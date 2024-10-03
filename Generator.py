from retrying import retry
import openai

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = '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'

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

        # print(reply)

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

        print(reply)

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
