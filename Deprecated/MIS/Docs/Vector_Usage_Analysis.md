# Vector-Based Semantic Comparison Usage

Modules using vector embeddings for similarity:

1. **auto_memory_filter.py**
   - Implements vector similarity scoring
   - Uses cosine distance between memory embeddings
   - Threshold: 0.75 similarity score

2. **neural_recall.py**
   - Employs BERT-based semantic clustering
   - Hybrid architecture with:
     - Transformer embeddings
     - Neural attention mechanisms
   - Similarity metrics:
     - Jaccard + SequenceMatcher hybrid
     - Threshold: 0.85 similarity

Non-vector modules:
- auto_memory_1.py (string matching)
- auto_memory_2.py (temporal context rules)
- enhanced_auto_memory_manager.py (cognitive weighting)