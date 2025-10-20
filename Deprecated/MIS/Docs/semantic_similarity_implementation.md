# Semantic Similarity and Embedding Implementation Guide

This document provides a detailed analysis of how semantic similarity and embedding techniques are implemented in various memory management modules for Open WebUI. The goal is to provide sufficient detail to implement these features in modules that don't currently have them.

## Table of Contents

1. [Overview of Semantic Similarity and Embeddings](#overview)
2. [Implementation in intelligent_llm.py](#intelligent-llm)
3. [Implementation in neural_recall.py](#neural-recall)
4. [Implementation in auto_memory_retrieval_and_storage.py](#auto-memory-retrieval)
5. [Practical Implementation Guide](#implementation-guide)
6. [Integration with Open WebUI](#integration)
7. [Optimization Techniques](#optimization)

<a name="overview"></a>
## 1. Overview of Semantic Similarity and Embeddings

Semantic similarity refers to the degree of relatedness between two pieces of text based on their meaning rather than just lexical overlap. Embeddings are vector representations of text that capture semantic meaning, allowing for efficient similarity calculations.

**Key Concepts:**

- **Embeddings**: Dense vector representations of text that capture semantic meaning
- **Vector Similarity**: Methods to calculate similarity between vectors (cosine similarity, dot product, etc.)
- **Vector Databases**: Storage systems optimized for vector similarity searches
- **Hybrid Approaches**: Combining embedding-based and LLM-based similarity for better results

<a name="intelligent-llm"></a>
## 2. Implementation in intelligent_llm.py

The `intelligent_llm.py` module implements a sophisticated memory system with embeddings through the `LongTermMemory` class (translated from `MemoireLongTerme`).

### 2.1 Embedding Generation

```python
def _generate_embedding(self, content: Any) -> np.ndarray:
    """Generates an embedding vector for the content."""
    # Simple example implementation
    if isinstance(content, str):
        return np.random.rand(256)  # Simple example
    return np.zeros(256)
```

While this implementation uses a placeholder (random vectors), it demonstrates the interface for embedding generation. In a real implementation, this would use a model like Word2Vec, BERT, or OpenAI's embedding API.

### 2.2 Memory Storage with Embeddings

```python
def store_memory(
    self,
    content: Any,
    memory_type: str,
    importance: float = 1.0,
    metadata: Dict = None,
):
    """Stores a new memory in the long-term memory."""
    vector = self._generate_embedding(content)
    with self.conn:
        cursor = self.conn.execute(
            """
            INSERT INTO memory (content, type, importance, timestamp, metadata, vector_embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                json.dumps(content),
                memory_type,
                importance,
                datetime.now().isoformat(),
                json.dumps(metadata or {}),
                pickle.dumps(vector),
            ),
        )
        memory_id = cursor.lastrowid
        self._update_index(memory_id, content, vector)
```

This method stores memories along with their vector embeddings in a SQLite database. The embedding is serialized using `pickle.dumps()`.

### 2.3 Similarity Calculation and Retrieval

```python
def _update_connections(self, new_memory_id: int, vector: np.ndarray):
    """Updates connections between memories based on similarity."""
    for other_id, other_vector in self.semantic_index.items():
        if other_id != new_memory_id:
            similarity = np.dot(vector, other_vector)
            if similarity > 0.5:  # Similarity threshold
                self.connection_strengths[(new_memory_id, other_id)] = similarity
                with self.conn:
                    self.conn.execute(
                        """
                        INSERT INTO connections (source_id, destination_id, strength, type)
                        VALUES (?, ?, ?, ?)
                    """,
                        (new_memory_id, other_id, similarity, "similarity"),
                    )

def retrieve_memory(self, content: Any) -> Optional[Any]:
    """Returns the memory closest to the provided content."""
    vector = self._generate_embedding(content)
    distances = [
        (id, np.dot(vector, other_vector))
        for id, other_vector in self.semantic_index.items()
    ]
    distances.sort(key=lambda x: x[1], reverse=True)
    if distances:
        return json.loads(
            self.conn.execute(
                """
            SELECT content FROM memory WHERE id = ?
        """,
                (distances[0][0],),
            ).fetchone()[0]
        )
    return None
```

These methods demonstrate:
1. Using dot product for similarity calculation
2. Storing similarity connections in a database
3. Using a similarity threshold (0.5) to filter relevant memories
4. Sorting by similarity score to retrieve the most relevant memories

<a name="neural-recall"></a>
## 3. Implementation in neural_recall.py

The `neural_recall.py` module implements a more sophisticated approach to similarity detection, particularly for deduplication.

### 3.1 Similarity Calculation for Deduplication

```python
def _calculate_memory_similarity(self, memory1: str, memory2: str) -> float:
    """
    Calculate similarity between two memory contents using a more robust method.
    Returns a score between 0.0 (completely different) and 1.0 (identical).
    """
    if not memory1 or not memory2:
        return 0.0
        
    # Clean the memories - remove tags and normalize
    memory1_clean = re.sub(r'\[Tags:.*?\]\s*', '', memory1).lower().strip()
    memory2_clean = re.sub(r'\[Tags:.*?\]\s*', '', memory2).lower().strip()
    
    # Handle exact matches quickly
    if memory1_clean == memory2_clean:
        return 1.0
        
    # Handle near-duplicates with same meaning but minor differences
    # Split into words and compare overlap
    words1 = set(re.findall(r'\b\w+\b', memory1_clean))
    words2 = set(re.findall(r'\b\w+\b', memory2_clean))
    
    if not words1 or not words2:
        return 0.0
        
    # Calculate Jaccard similarity for word overlap
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    jaccard = intersection / union if union > 0 else 0.0
    
    # Use sequence matcher for more precise comparison
    seq_similarity = SequenceMatcher(None, memory1_clean, memory2_clean).ratio()
    
    # Combine both metrics, weighting sequence similarity higher
    combined_similarity = (0.4 * jaccard) + (0.6 * seq_similarity)
    
    return combined_similarity
```

This method demonstrates a hybrid approach to similarity calculation:
1. Text normalization (removing tags, lowercasing)
2. Jaccard similarity for word overlap
3. Sequence matching for more precise comparison
4. Weighted combination of both metrics

### 3.2 Using Similarity for Memory Deduplication

```python
# Check each new memory against existing ones
for memory_dict in memories:
    if memory_dict["operation"] == "NEW":
        # Format the memory content
        operation = MemoryOperation(**memory_dict)
        formatted_content = self._format_memory_content(operation)
        
        # Check for similarity with existing memories
        is_duplicate = False
        for existing_content in existing_contents:
            similarity = self._calculate_memory_similarity(
                formatted_content, existing_content
            )
            if similarity >= self.valves.similarity_threshold:
                logger.debug(
                    f"Skipping duplicate memory (similarity: {similarity:.2f}): {formatted_content[:50]}..."
                )
                is_duplicate = True
                break
                
        if not is_duplicate:
            processed_memories.append(memory_dict)
```

This code shows how similarity is used for deduplication:
1. Calculate similarity between new and existing memories
2. Use a configurable threshold to determine duplicates
3. Skip memories that are too similar to existing ones

<a name="auto-memory-retrieval"></a>
## 4. Implementation in auto_memory_retrieval_and_storage.py

This module uses a hybrid approach, combining LLM-based relevance scoring with some vector-based techniques.

### 4.1 Memory Relevance Analysis

```python
async def get_relevant_memories(
    self,
    current_message: str,
    user_id: str,
):
    """Get memories relevant to the current context using OpenAI."""
    try:
        # Get existing memories
        existing_memories = Memories.get_memories_by_user_id(user_id=str(user_id))
        
        # Create prompt for memory relevance analysis
        memory_prompt = f"""Given the current user message: "{current_message}"

Please analyze these existing memories and select the all relevant ones for the current context.
Better to err on the side of including too many memories than too few.
Consider what information is needed to answer the question, location or habits information is often relevant for answering questions.
Rate each memory's relevance from 0-10 and explain why it's relevant.

Available memories:
{memory_contents}

Return the response in this exact JSON format without any extra newlines:
[{{"memory": "exact memory text", "relevance": score, "id": "id of the memory"}}, ...]
"""

        # Get OpenAI's analysis
        response = await self.query_openai_api(self.valves.model, memory_prompt, current_message)
        
        try:
            # Clean response and parse JSON
            cleaned_response = response.strip().replace("\n", "").replace("    ", "")
            memory_ratings = json.loads(cleaned_response)
            relevant_memories = [item["memory"] for item in sorted(memory_ratings, key=lambda x: x["relevance"], reverse=True) if item["relevance"] >= 5][
                : self.valves.related_memories_n
            ]

            return relevant_memories

        except json.JSONDecodeError as e:
            print(f"Failed to parse OpenAI response: {e}\n")
            return []

    except Exception as e:
        print(f"Error getting relevant memories: {e}\n")
        return []
```

This implementation uses an LLM to rate memory relevance rather than vector similarity, but demonstrates the concept of relevance scoring and filtering.

<a name="implementation-guide"></a>
## 5. Practical Implementation Guide

Based on the analyzed modules, here's a practical guide to implementing semantic similarity and embeddings in a memory management module.

### 5.1 Embedding Generation

```python
async def generate_embedding(self, text: str) -> List[float]:
    """
    Generate an embedding vector for the given text.
    
    Args:
        text: The text to generate an embedding for
        
    Returns:
        A list of floats representing the embedding vector
    """
    # Option 1: Use OpenAI's embedding API
    if self.valves.embedding_provider == "openai":
        try:
            url = f"{self.valves.openai_api_url}/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.valves.openai_api_key}",
            }
            payload = {
                "model": "text-embedding-ada-002",  # or another embedding model
                "input": text,
            }
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["data"][0]["embedding"]
                
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            # Fall back to a simpler method
            
    # Option 2: Use a local model via Ollama
    elif self.valves.embedding_provider == "ollama":
        try:
            url = f"{self.valves.ollama_api_url.rstrip('/')}/api/embeddings"
            payload = {
                "model": self.valves.ollama_embedding_model,
                "prompt": text,
            }
            
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["embedding"]
                
        except Exception as e:
            logger.error(f"Error generating Ollama embedding: {e}")
            # Fall back to a simpler method
    
    # Option 3: Simple fallback using hashing (not recommended for production)
    import hashlib
    import struct
    
    # Generate a simple hash-based embedding (for demonstration only)
    hash_values = []
    for i in range(0, len(text), 3):
        chunk = text[i:i+3]
        hash_val = int(hashlib.md5(chunk.encode()).hexdigest(), 16)
        # Convert to a float between -1 and 1
        float_val = (hash_val % 10000) / 5000 - 1
        hash_values.append(float_val)
    
    # Pad or truncate to get a consistent size
    target_size = 128
    if len(hash_values) < target_size:
        hash_values.extend([0.0] * (target_size - len(hash_values)))
    else:
        hash_values = hash_values[:target_size]
        
    return hash_values
```

### 5.2 Similarity Calculation

```python
def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate the similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        A float between 0.0 and 1.0 representing similarity
    """
    # Option 1: Cosine similarity
    import numpy as np
    
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    cosine_similarity = dot_product / (norm1 * norm2)
    
    # Normalize to 0-1 range (cosine similarity is between -1 and 1)
    return (cosine_similarity + 1) / 2
```

### 5.3 Memory Storage with Embeddings

```python
async def store_memory_with_embedding(self, content: str, user_id: str) -> str:
    """
    Store a memory with its embedding vector.
    
    Args:
        content: The memory content
        user_id: The user ID
        
    Returns:
        The memory ID
    """
    try:
        # Generate embedding
        embedding = await self.generate_embedding(content)
        
        # Store in Open WebUI memory system
        memory_obj = await add_memory(
            request=Request(scope={"type": "http", "app": webui_app}),
            form_data=AddMemoryForm(
                content=content,
                metadata=json.dumps({"embedding": embedding})
            ),
            user=Users.get_user_by_id(user_id),
        )
        
        return memory_obj.id
        
    except Exception as e:
        logger.error(f"Error storing memory with embedding: {e}")
        # Fall back to storing without embedding
        memory_obj = await add_memory(
            request=Request(scope={"type": "http", "app": webui_app}),
            form_data=AddMemoryForm(content=content),
            user=Users.get_user_by_id(user_id),
        )
        
        return memory_obj.id
```

### 5.4 Memory Retrieval with Embeddings

```python
async def get_relevant_memories_with_embeddings(
    self, query: str, user_id: str, threshold: float = 0.7, max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Get memories relevant to the query using embedding similarity.
    
    Args:
        query: The query text
        user_id: The user ID
        threshold: Minimum similarity threshold
        max_results: Maximum number of results
        
    Returns:
        List of relevant memories with similarity scores
    """
    try:
        # Generate query embedding
        query_embedding = await self.generate_embedding(query)
        
        # Get all memories for the user
        all_memories = Memories.get_memories_by_user_id(user_id)
        
        # Calculate similarity for each memory
        memory_similarities = []
        for memory in all_memories:
            try:
                # Extract embedding from metadata if available
                metadata = json.loads(memory.metadata) if hasattr(memory, "metadata") and memory.metadata else {}
                memory_embedding = metadata.get("embedding")
                
                # If embedding is not available, generate it
                if not memory_embedding:
                    memory_embedding = await self.generate_embedding(memory.content)
                    
                    # Update memory with embedding
                    # (This would require a method to update memory metadata)
                
                # Calculate similarity
                similarity = self.calculate_similarity(query_embedding, memory_embedding)
                
                # Add to results if above threshold
                if similarity >= threshold:
                    memory_similarities.append({
                        "id": memory.id,
                        "content": memory.content,
                        "similarity": similarity,
                    })
            except Exception as e:
                logger.error(f"Error processing memory {memory.id}: {e}")
                continue
                
        # Sort by similarity (descending) and limit results
        memory_similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return memory_similarities[:max_results]
        
    except Exception as e:
        logger.error(f"Error retrieving memories with embeddings: {e}")
        return []
```

### 5.5 Hybrid Approach (Combining Embeddings and LLM)

```python
async def get_relevant_memories_hybrid(
    self, query: str, user_id: str, max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Get memories relevant to the query using a hybrid approach.
    
    Args:
        query: The query text
        user_id: The user ID
        max_results: Maximum number of results
        
    Returns:
        List of relevant memories with combined scores
    """
    try:
        # Step 1: Get initial candidates using embedding similarity (faster)
        embedding_candidates = await self.get_relevant_memories_with_embeddings(
            query, user_id, threshold=0.5, max_results=max_results * 2
        )
        
        if not embedding_candidates:
            return []
            
        # Step 2: Refine with LLM-based scoring (more accurate)
        candidate_contents = [mem["content"] for mem in embedding_candidates]
        formatted_memories = "\n".join([f"- {mem}" for mem in candidate_contents])
        
        system_prompt = f"""
        You are a memory relevance evaluator. Your task is to rate how relevant each memory is to the current query.
        
        Query: "{query}"
        
        Memories:
        {formatted_memories}
        
        Rate each memory's relevance from 0.0 to 1.0, where:
        - 0.0 means completely irrelevant
        - 1.0 means highly relevant
        
        Return your ratings as a JSON array of objects with 'memory' and 'score' properties.
        """
        
        # Get LLM ratings
        llm_response = await self.query_llm(system_prompt, "")
        
        try:
            llm_ratings = json.loads(self._clean_json_response(llm_response))
            
            # Step 3: Combine embedding and LLM scores
            combined_scores = []
            for emb_mem in embedding_candidates:
                # Find matching LLM rating
                llm_rating = next(
                    (r for r in llm_ratings if r["memory"] == emb_mem["content"]),
                    None
                )
                
                if llm_rating:
                    # Combine scores (weighted average)
                    combined_score = (
                        emb_mem["similarity"] * 0.4 +  # 40% weight to embedding
                        llm_rating["score"] * 0.6      # 60% weight to LLM
                    )
                    
                    combined_scores.append({
                        "id": emb_mem["id"],
                        "content": emb_mem["content"],
                        "score": combined_score,
                    })
                else:
                    # Use embedding score only
                    combined_scores.append({
                        "id": emb_mem["id"],
                        "content": emb_mem["content"],
                        "score": emb_mem["similarity"],
                    })
                    
            # Sort by combined score and limit results
            combined_scores.sort(key=lambda x: x["score"], reverse=True)
            return combined_scores[:max_results]
            
        except json.JSONDecodeError:
            # Fall back to embedding results if LLM parsing fails
            return embedding_candidates[:max_results]
            
    except Exception as e:
        logger.error(f"Error in hybrid memory retrieval: {e}")
        return []
```

<a name="integration"></a>
## 6. Integration with Open WebUI

To integrate semantic similarity and embeddings with Open WebUI, you need to:

1. **Store Embeddings**: Store embeddings as metadata in the memory objects
2. **Update Memory Schema**: Ensure the memory schema can accommodate embedding data
3. **Handle API Calls**: Implement proper API calls for embedding generation
4. **Manage Caching**: Cache embeddings to avoid regenerating them frequently

### 6.1 Memory Storage Integration

```python
# Example of storing a memory with embedding metadata
async def add_memory_with_embedding(content: str, user: Any) -> Any:
    """Add a memory with embedding metadata."""
    # Generate embedding
    embedding = await generate_embedding(content)
    
    # Create metadata with embedding
    metadata = {
        "embedding": embedding,
        "embedding_model": "text-embedding-ada-002",  # or whatever model you're using
        "embedding_version": "1.0",
        "created_at": datetime.now().isoformat(),
    }
    
    # Add memory with metadata
    memory_obj = await add_memory(
        request=Request(scope={"type": "http", "app": webui_app}),
        form_data=AddMemoryForm(
            content=content,
            metadata=json.dumps(metadata)
        ),
        user=user,
    )
    
    return memory_obj
```

<a name="optimization"></a>
## 7. Optimization Techniques

Several optimization techniques can improve the performance and efficiency of embedding-based memory systems:

### 7.1 Caching

```python
class EmbeddingCache:
    """Simple cache for embeddings to avoid regenerating them."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        
    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
        
    def set(self, key: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        # Evict least recently used item if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            
        self.cache[key] = embedding
        self.access_times[key] = time.time()
```

### 7.2 Batch Processing

```python
async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts in a single API call."""
    if not texts:
        return []
        
    try:
        url = f"{self.valves.openai_api_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.openai_api_key}",
        }
        payload = {
            "model": "text-embedding-ada-002",
            "input": texts,
        }
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return [item["embedding"] for item in data["data"]]
            
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        # Fall back to individual generation
        results = []
        for text in texts:
            try:
                embedding = await self.generate_embedding(text)
                results.append(embedding)
            except Exception:
                results.append([0.0] * 1536)  # Default embedding size for ada-002
        return results
```

### 7.3 Dimensionality Reduction

For very large embedding collections, you can use dimensionality reduction techniques:

```python
def reduce_embedding_dimensions(self, embedding: List[float], target_dim: int = 128) -> List[float]:
    """Reduce the dimensionality of an embedding vector."""
    import numpy as np
    from sklearn.decomposition import PCA
    
    # Convert to numpy array
    vec = np.array(embedding).reshape(1, -1)
    
    # Initialize PCA if not already done
    if not hasattr(self, 'pca') or self.pca.n_components != target_dim:
        self.pca = PCA(n_components=target_dim)
        # Note: In a real implementation, you would fit the PCA on a large dataset first
        
    # Transform the vector
    reduced_vec = self.pca.transform(vec)[0]
    
    return reduced_vec.tolist()
```

## Conclusion

Implementing semantic similarity and embeddings in a memory management module involves:

1. **Generating Embeddings**: Using an API or local model to create vector representations
2. **Calculating Similarity**: Using vector operations to determine relevance
3. **Storing and Retrieving**: Managing embeddings alongside memory content
4. **Hybrid Approaches**: Combining embedding-based and LLM-based methods
5. **Optimization**: Using caching, batching, and other techniques for efficiency

By following the implementation patterns described in this document, you can add semantic similarity capabilities to any memory management module, improving memory retrieval relevance and efficiency.