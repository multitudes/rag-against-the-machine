what are chunks? 
Using the `dir` attribute i can print out the structure and params of the obj:

INFO:chunking:Chunk object attributes: ['__annotations__', '__class__', '__dataclass_fields__', '__dataclass_params__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__match_args__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_preview_embedding', 'context', 'copy', 'embedding', 'end_index', 'from_dict', 'id', 'start_index', 'text', 'to_dict', 'token_count']
so it seems i can get to print these 
'context', 'copy', 'embedding', 'end_index', 'from_dict', 'id', 'start_index', 'text', 'to_dict', 'token_count']

So to display the params for debug:
```python
    chunk_dict = first_chunk.to_dict()
    pretty_json = json.dumps(chunk_dict, indent=4)
    logger.info(f"Chunk as pretty dictionary:\n{pretty_json}")
```
