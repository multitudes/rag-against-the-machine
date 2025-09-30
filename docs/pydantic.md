# pydantic

**Pydantic v1 vs v2 vs v3:**

- **Pydantic v1**: The original version. Uses Python type hints for data validation and parsing. Models are based on standard Python classes. Validation is performed at runtime. Widely used and stable, but slower and less flexible for advanced use cases.

- **Pydantic v2**: Major rewrite for performance and flexibility. Uses a new core written in Rust for much faster validation. The API is more strict and explicit. Some features and syntax have changed (e.g., `model_validate` instead of `parse_obj`). Migration from v1 to v2 may require code changes.

- **Pydantic v3**: Latest version (still in development as of mid-2025). Builds on v2, with further improvements, stricter typing, and new features. May introduce breaking changes and new APIs.

**Summary:**  
- All versions provide data validation using Python type hints.
- v2 and v3 are faster and more strict than v1.
- If your project requires stability and compatibility, use v1.  
- If you want better performance and modern features, use v2 or v3 (but check your code for compatibility).

**Docs:**  
- [Pydantic v1](https://docs.pydantic.dev/1.10/)
- [Pydantic v2](https://docs.pydantic.dev/latest/)
- [Pydantic v3](https://docs.pydantic.dev/dev/)

## Yes, you can use Pydantic to define classes for your request and response data.  
Pydantic provides type validation and automatic conversion, making your code safer and easier to maintain.

**Why use Pydantic?**
- Ensures your data matches the expected types and structure.
- Automatically validates and parses data from JSON or dicts.
- Makes your code more readable and robust.

**Example: Define a request model with Pydantic**

```python
from pydantic import BaseModel
from typing import List, Dict, Any

class Message(BaseModel):
    role: str
    content: str

class FunctionParameter(BaseModel):
    type: str
    properties: Dict[str, Any]
    required: List[str]

class Function(BaseModel):
    name: str
    description: str
    parameters: FunctionParameter

class Tool(BaseModel):
    type: str
    function: Function

class RequestData(BaseModel):
    model: str
    messages: List[Message]
    tools: List[Tool]
    stream: bool = False
    think: bool = False

# Usage
request_data = RequestData(
    model="qwen3:0.6b",
    messages=[Message(role="user", content="What is 2 + 2?")],
    tools=tools,  # assuming tools is a list of Tool objects
    stream=False,
    think=False
)
```

**Benefits:**
- If you get data from a file or user input, Pydantic will validate it.
- You can easily convert between dicts and Pydantic models (`.dict()`, `.json()`).
- If the data is invalid, Pydantic raises a clear error.

**Summary:**  
Use Pydantic classes for request/response data to ensure correctness, validation, and maintainability—especially when working with structured data from LLMs or APIs.


## resources

https://pypi.org/project/pydantic/