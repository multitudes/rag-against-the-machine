# Pydantic in this project

The subject requires **Pydantic** for the data we exchange between
stages (questions, sources, search results). We use Pydantic v2
(`model_validate`, `model_dump`, `model_dump_json`). Service classes
(`Searcher`, the CLI) are plain Python.

Models live in `src/core/schemas.py`. They match the moulinette JSON:
`MinimalSource`, `UnansweredQuestion`, `MinimalSearchResults`,
`StudentSearchResults`, and the answer variants.

## Why we use it

Incoming JSON is checked against types and required fields. A missing
`question` or a bad offset fails fast instead of producing a file the
grader rejects. `model_dump_json(indent=2)` is what we write to disk
for `search_dataset` / `answer_dataset`.

## Example from our schemas

```python
class UnansweredQuestion(BaseModel):
    """An unanswered question."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
```

`question` is required. `question_id` is generated when we construct
the model without one (single `search` / `answer` on the CLI).

We create instances from JSON with `model_validate_json`, not the
v1 `parse_obj` API.

## References

- [Pydantic v2](https://docs.pydantic.dev/latest/)
