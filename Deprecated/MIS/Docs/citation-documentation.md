# Using Citations for Process Documentation in Open WebUI Modules

This document summarizes how the SMART module uses the citation field to document its internal processes, providing a pattern that can be implemented in other modules.

## Citation System Implementation

### Core Function

The SMART module implements a citation system through a factory function that creates a citation sender:

```python
def get_send_citation(__event_emitter__: EmitterType) -> SendCitationType:
    async def send_citation(url: str, title: str, content: str):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": False}],
                    "source": {"name": title},
                },
            }
        )
    return send_citation
```

### Event Structure

Citations use a consistent JSON structure:
- `type`: Always "citation" to identify the event type
- `data`: Contains the citation content
  - `document`: Array containing the content text
  - `metadata`: Array of metadata objects with source URL and HTML flag
  - `source`: Object with the citation name/title

## Usage Patterns

The SMART module uses citations for several distinct purposes:

### 1. Documenting Agent Outputs

```python
await send_citation(
    url=f"SMART Planning",
    title="SMART Planning",
    content=f"{content=}",
)
```

This pattern is used to document the raw outputs from different agents in the pipeline, including:
- Planning agent decisions
- Reasoning agent thought processes
- Tool agent responses

### 2. Logging Tool Usage

```python
await send_citation(
    url=f"Tool call {num_tool_calls}",
    title=event["name"],
    content=f"Tool '{event['name']}' with inputs {data.get('input')} returned {data.get('output')}",
)
```

This pattern creates a detailed log of tool interactions, capturing:
- Tool name
- Input parameters
- Output results
- Sequential numbering of tool calls

### 3. Process Transparency

Citations provide transparency into the module's internal processes by exposing:
- Decision-making logic
- Intermediate steps
- Raw agent outputs before formatting

## Implementation Strategy

To implement a similar citation system in another module:

1. **Create the Citation Function**:
   ```python
   def get_send_citation(event_emitter):
       async def send_citation(url, title, content):
           if event_emitter is None:
               return
           await event_emitter({
               "type": "citation",
               "data": {
                   "document": [content],
                   "metadata": [{"source": url, "html": False}],
                   "source": {"name": title},
               },
           })
       return send_citation
   ```

2. **Initialize in Your Module**:
   ```python
   send_citation = get_send_citation(__event_emitter__)
   ```

3. **Document Key Process Steps**:
   ```python
   # Document initialization
   await send_citation(
       url="module://initialization",
       title="Module Initialization",
       content=f"Configuration: {config}"
   )
   
   # Document processing stages
   await send_citation(
       url="module://processing/stage1",
       title="Processing Stage 1",
       content=f"Intermediate results: {results}"
   )
   
   # Document external API calls
   await send_citation(
       url=f"api://{api_name}/call/{call_id}",
       title=f"{api_name} API Call",
       content=f"Request: {request}\nResponse: {response}"
   )
   ```

## Best Practices

1. **Use Consistent URL Schemes**:
   - Create a URL scheme that categorizes different types of citations
   - Example: `module://category/subcategory`

2. **Descriptive Titles**:
   - Use clear, concise titles that identify the citation purpose
   - Include identifiers for sequential operations

3. **Structured Content**:
   - Format content consistently for easier parsing
   - Include relevant variables and their values
   - Consider using JSON formatting for complex data

4. **Strategic Placement**:
   - Add citations at key decision points
   - Document inputs and outputs of significant operations
   - Log external interactions

5. **Error Documentation**:
   - Use citations to document errors and exceptions
   - Include error context and troubleshooting information

By following these patterns, you can implement a citation system that provides transparency into your module's internal processes, facilitating debugging and user understanding.