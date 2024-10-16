# MMLF
The code implementation of paper MMLF: Multi-query Multi-passage Late Fusion Retrieval

---

## Step to run the code
### 1. Setup the env

- Python 3.x
- Required packages can be installed using:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Download the pregenerated subqueries and queries from [huggingface](https://huggingface.co/datasets/yvonne90190/MMLF_Generated_Results/tree/main)

### 3. Set the `OPENAI_KEY` in both `config.py`

---

## Script

### 1. `Generate.py`

```bash
python Generate.py --generation_stage={} --generation_type={} --task={} --queries_file={} --passages_file={}
```
- **`generation_stage`**: The generation stage you want to use. Candidate values:
  - `first`: First stage of query generation.
  - `second`: Second stage of query generation.
  - `combined`: Combines both stages.
  
- **`generation_type`**: The query generation method name. Candidate values:
  - `CoT`: Chain of Thought generation.
  - `Q2D`: Query to Document.
  - `MQR`: Multi Query Refinement.
  - `MCQE`: Multi Candidate Query Expansion.
  - `QQD`: Query-Query Document generation.
  
- **`task`**: The dataset you want to use. Candidate values: `trec-covid`, `fiqa`, `dbpedia-entity`, `nfcorpus`, `webis-touche2020`.
  
- **`queries_file`**: The path to the queries file (optional).
  
- **`passages_file`**: The path to the passages file (optional).

---

### 2. `Retrieve.py`
This script retrieves passages using queries processed through the MMLF method. It supports single or multiple query retrieval with optional query expansion and result fusion.

```bash
python Retrieve.py --retrieval_type={} --fusion_method={} --include_original --concat_original --base_model={} --task={} --queries_file={} --result_file={}
```

- **`retrieval_type`**: The retrieval type you want to use. Candidate values:
  - `single`: Uses a single query for retrieval.
  - `multiple`: Uses multiple expanded queries.
  - `rawQ`: Uses raw queries without expansion.
  
- **`fusion_method`**: The fusion method used to combine results from multiple queries. Candidate values:
  - `RRF`: Reciprocal Rank Fusion.
  - `None`: For single query retrieval.
  
- **`include_original`**: A flag that, if set, includes the original queries in the retrieval process.
  
- **`concat_original`**: A flag that, if set, concatenates original queries with expanded queries.
  
- **`base_model`**: The base model to use for query encoding. Candidate values: `e5-small-v2`, `contriever`, or other models.
  
- **`task`**: The dataset you want to use. Candidate values: `trec-covid`, `fiqa`, `dbpedia-entity`, `nfcorpus`, `webis-touche2020`.
  
- **`queries_file`**: The path to the queries file.
  
- **`result_file`**: The path to the file where the results will be saved.

---

### Example

For `MMLF`:
```bash
python Generate.py --generation_stage first --generation_type MQR --task "$task_type" --queries_file "your_queries_file"
python Generate.py --generation_stage second --generation_type CQE --task "$task_type" --queries_file "$task_type" --queries_file "your_queries_file" --passages_file "your_paggages_file"
python Retrieve.py --retrieval_type multiple  --fusion_method RRF --include_original --base_model "$base_model" --task "$task_type" --queries_file "your_paggages_file" --result_file "your_result_file"
```

---

## License
This project is licensed under the xxx License.

