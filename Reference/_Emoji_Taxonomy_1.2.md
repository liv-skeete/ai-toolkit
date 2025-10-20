# AMM Emoji Taxonomy v1.2 (Pictorial Citation Format)

memory_lifecycle:       # Used for status emissions
  retrieving: 💭        # Recalling operation, eg. LLM retreiving
  identifying: 🧠       # Thinking operation, eg. LLM reranking
  storing: 💾           # Writing, updating, or saving information
  deleting: 🗑️          # Delete operation
  updating: ♻️          # Update Operation
  purging: 🔥           # Purging or pruning Operations
  no_match: 🚫          # No match found
  no_work: 🤷‍♂️           # No mork required
  complete: 🏁          # Processing complete
  up_to_date: ✅        # Information is current
  error: ⚠️             # Processing error
  important:❗️          # Important event
  cleanup: 🧹           # Clean-up operations
  chatting: 💬          # Chatting
  choosing: 🤔          # System is choosing best option
  communicating: 🛜 

memory_bullets:  # Used for citation listings
  read: 🔍              # Retrieved and reviewed in context
  created: 💾           # Retained: memory was committed
  deleted: 🗑️           # Memory entry removed (auto or user-driven)
  updated: ♻️           # Meemory entry updated
  purged: 🔥            # Memory entry removed by pruning process
  deduped: ♻️           # Meemory entry consolidated by dedupe process

# Notes:
# - This visual language mirrors system-level taxonomy for consistency across logs/UI