# Semantic Memory Retrieval Implementation Summary

## Project Overview

We've been working on implementing semantic memory retrieval in the Automatic Memory Manager (AMM) modules for Open WebUI. The goal is to use the `/api/v1/memories/query` API endpoint to retrieve memories based on semantic similarity rather than using direct embedding code.

## What We've Learned

### API Structure

- The `/api/v1/memories/query` endpoint provides semantic search functionality
- It requires authentication with a JWT token in the Authorization header
- It returns memories with their semantic distances (lower = more similar)
- The endpoint accepts parameters:
  - `content`: The text to search for similar memories
  - `k`: The number of memories to retrieve

### Authentication Challenges

- JWT tokens are user-specific and contain the user ID in the payload
- The token is required for authentication but is not directly available in the user data passed to the module
- We successfully tested with a hardcoded token, confirming the API works when properly authenticated
- We've explored but haven't yet resolved how to obtain the correct token for each user

### Authentication Endpoint

- The `/api/v1/auths/signin` endpoint can be used to obtain JWT tokens
- It requires email and password credentials:
  ```bash
  curl -X POST http://localhost:3000/api/v1/auths/signin \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your_email@example.com",
    "password": "your_password"
  }'
  ```
- The response includes the JWT token and user information:
  ```json
  {
    "token": "your_jwt_token",
    "token_type": "Bearer",
    "id": "user_id",
    "email": "your_email@example.com",
    "name": "your_name",
    "role": "your_role",
    "profile_image_url": "your_profile_image_url"
  }
  ```
- This endpoint authenticates as the user whose credentials are provided
- It doesn't allow getting a token for a different user ID

### Implementation Progress

1. **What Works**:
   - API connection with proper base URL
   - Processing API responses to extract memories and their scores
   - Converting distance values to scores (1.0 - distance)
   - Filtering results based on a semantic threshold
   - Formatting memories for inclusion in the context

2. **What Doesn't Work**:
   - Direct VECTOR_DB_CLIENT access (runs in a different context)
   - Extracting tokens from user data (tokens not present)
   - Using a hardcoded token for all users (causes incorrect context insertion)

3. **Current Implementation**:
   - Detailed logging of request object to find potential token sources
   - Skip API query if no token is available
   - Fall back to LLM-based scoring when API query fails

## Current Status

- The module is functional but limited by the authentication challenge
- We've removed the hardcoded token to prevent incorrect context insertion
- We're investigating options for obtaining user-specific tokens
- The code is structured to easily integrate a token solution when available

## Next Steps

1. **Authentication Solution**:
   - Research how to obtain user-specific tokens
   - Explore if there's a way to generate tokens based on user ID
   - Consider if there's an admin or service account approach

2. **Potential Approaches**:
   - Using the `/api/v1/auths/signin` endpoint with appropriate credentials
   - Token lookup service at session start
   - Integration with authentication middleware
   - Custom token generation if secret key is available

3. **Implementation Refinement**:
   - Add token caching for performance
   - Improve error handling and logging
   - Add configuration options for token management

## Technical Details

- The API returns a structure with ids, documents, metadatas, and distances
- We convert distances to scores (1.0 - distance) for compatibility with existing code
- The semantic threshold is configurable via the valve parameters
- The module falls back to LLM-based scoring if the API query fails

## Direct Vector Database Access (2025-03-15 Update)

We explored an alternative approach to semantic memory retrieval by directly accessing the vector database instead of using the API endpoint. This would bypass the authentication challenges we're facing with the API.

### What We Did

1. Added test code to check if VECTOR_DB_CLIENT is accessible from the module
2. Inspected the search method signature and parameters
3. Explored the embedding function parameters and requirements
4. Implemented a test that generates real embeddings and searches for similar memories

### What Worked

1. **Vector Database Access**: Successfully imported and accessed VECTOR_DB_CLIENT
2. **Embedding Generation**: Successfully generated embeddings using the correct parameters:
   ```
   embedding_function = get_embedding_function(
       embedding_engine="openai",
       embedding_model="text-embedding-ada-002",
       embedding_function=None,
       url="https://api.openai.com/v1",
       key=self.valves.openai_api_key,
       embedding_batch_size=1
   )
   ```
3. **Search Method**: Successfully called the search method with the correct parameters:
   ```
   VECTOR_DB_CLIENT.search(
       collection_name=collection_name,
       vectors=[query_embedding],
       limit=10
   )
   ```

### What Didn't Work

1. **Search Results**: The search method returned no results (`None`) even though:
   - The collection exists and contains memories
   - We successfully generated embeddings
   - We used the correct search parameters

### Next Steps

1. **Implement Direct Query Method**:
   - Create a `_query_memories_direct` method that uses the vector database directly
   - Use the correct embedding function parameters we discovered
   - Handle the case where search returns no results
   - Format the results to match our existing memory format

2. **Investigate Search Issues**:
   - Check if the collection has proper embeddings
   - Explore why the search returns no results
   - Test with different queries and parameters

3. **Integration**:
   - Integrate the direct query method into the `get_relevant_memories` function
   - Add it as a fallback when the API query fails
   - Ensure proper error handling and logging

4. **Performance Optimization**:
   - Consider caching embeddings for common queries
   - Optimize the embedding generation process
   - Add configuration options for the direct query method