# pydantic

**Pydantic v1 vs v2 vs v3:**

- **Pydantic v1**: The original version. Uses Python type hints for data validation and parsing. Models are based on standard Python classes. Validation is performed at runtime. Widely used and stable, but slower and less flexible for advanced use cases.

- **Pydantic v2**: Major rewrite for performance and flexibility. Uses a new core written in Rust for much faster validation. The API is more strict and explicit. Some features and syntax have changed (e.g., `model_validate` instead of `parse_obj`). Migration from v1 to v2 may require code changes.

- **Pydantic v3**: Latest version (still in development as of mid-2025). Builds on v2, with further improvements, stricter typing, and new features. May introduce breaking changes and new APIs.

- All versions provide data validation using Python type hints.
- v2 and v3 are faster and more strict than v1.
- If your project requires stability and compatibility, use v1.  
- If you want better performance and modern features, use v2 or v3 (but check your code for compatibility).

**Docs:**  

- [Pydantic v1](https://docs.pydantic.dev/1.10/)
- [Pydantic v2](https://docs.pydantic.dev/latest/)
- [Pydantic v3](https://docs.pydantic.dev/dev/)

## example with defaults

```python
# Question Models
class UnansweredQuestion(BaseModel):
    """Represents an unanswered question."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

```

This code defines a **Pydantic data model** representing a question that hasn't been answered yet. It enforces typing and automatically generates a unique identifier for every new instance.

Here is a line-by-line breakdown of what each part means:

### 1. `class UnansweredQuestion(BaseModel):`

* Defines a new class that inherits from Pydantic’s `BaseModel`.
* Pydantic models automatically validate incoming data, parse types, and provide helpful methods like converting the object to a dictionary or JSON.

### 2. `"""Represents an unanswered question."""`

* A **docstring** explaining the purpose of the class. Pydantic uses this string when generating OpenAPI/FastAPI documentation.

### 3. `question_id: str = Field(...)`

* Defines a field named `question_id` that must be a string (`str`).
* `Field(...)` is used to add metadata or special default behaviors to a Pydantic field.

### 4. `default_factory=lambda: str(uuid.uuid4())`

* **`uuid.uuid4()`**: Generates a random, globally unique identifier (UUID v4).
* **`lambda: ...`**: A small anonymous function that calls `str(uuid.uuid4())`.
* **`default_factory`**: Instructs Pydantic to execute this function **every time a new `UnansweredQuestion` instance is created** without an explicit `question_id`. This ensures every question gets its own unique ID rather than sharing a single pre-generated one.

### 5. `question: str`

* Defines a required field named `question`.
* Since it has no default value, you **must** supply a string for this field whenever you create an instance (e.g., `UnansweredQuestion(question="What is Python?")`).

---

### Example Usage

```python
import uuid
from pydantic import BaseModel, Field


class UnansweredQuestion(BaseModel):
    """Represents an unanswered question."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


# Creating an instance (question_id is automatically generated)
q1 = UnansweredQuestion(question="How does Pydantic work?")

print(q1.question_id)
# Output: 'b3f12a84-72de-4e92-94b1-e2831f49fa82' (a random UUID)

print(q1.question)
# Output: 'How does Pydantic work?'

```

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


## resources

https://pypi.org/project/pydantic/