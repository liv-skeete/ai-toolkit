# Semantic Comparison Implementation Details

## 1. auto_memory_filter.py
**Embedding Generation**:
- External API calls via OpenAI/Ollama endpoints
- Custom implementation (not native Open WebUI):
  ```python
  async def query_openai_api():
      url = f"{self.valves.openai_api_url}/v1/embeddings"
      payload = {"input": text, "model": "text-embedding-ada-002"}
  ```

**Storage**:
- Direct DB writes using raw SQL:
  ```python
  cursor.execute("INSERT INTO embeddings (id, vector) VALUES (?, ?)", 
               (memory_id, pickle.dumps(embedding)))
  ```

**Comparison**:
- Cosine similarity calculation:
  ```python
  def cosine_similarity(vec1, vec2):
      return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
  ```

## 2. neural_recall.py
**Embedding Architecture**:
- Hybrid BERT+Transformer system
- Custom model endpoint integration:
  ```python
  self.bert_endpoint = "http://bert-service:7689/embed"
  async def get_bert_embedding(text):
      async with session.post(self.bert_endpoint, json={"text": text})...
  ```

**System Integration**:
- Dual-mode operation:
  1. API mode: For new memory processing
  2. Direct DB cache: For existing embeddings
  ```python
  if use_cached:
      return self.cache[text_hash]
  else:
      return await self.get_bert_embedding(text)
  ```

**Native Open WebUI Integration**:
```diff
- Does NOT use native OWUI embeddings
+ Implements separate vector pipeline:
  - Custom dimension (768 vs OWUI's 512)
  - Batch processing queue
  - GPU-accelerated inference
```

## Key Differences
| Aspect                | auto_memory_filter.py         | neural_recall.py            |
|-----------------------|-------------------------------|-----------------------------|
| Embedding Source      | External API                  | Custom BERT endpoint        |
| Storage Format        | Pickled numpy arrays          | Quantized FP16 vectors      |
| Similarity Metric     | Cosine                        | Hybrid Attention+Jaccard    |
| DB Interaction        | Raw SQL                       | ORM with connection pooling |
| Native Integration    | Partial (API config)          | None (fully custom)         |
| Update Frequency      | On write                      | Real-time inference         |