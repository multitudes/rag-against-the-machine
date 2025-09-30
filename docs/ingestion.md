**Ingestion** is the process of preparing and loading your data into your RAG system so it can be searched later. It's like building an index for a library - you need to organize all the books before people can find them.

## What Ingestion Does:

### 1. **Document Processing**
- Read files from the vLLM repository
- Extract text content from different file types (.py, .md, .txt, etc.)
- Clean and preprocess the text

### 2. **Chunking** 
- Break large documents into smaller pieces (chunks)
- Each chunk should be meaningful and searchable
- You're using `chonkie` library for this

### 3. **Indexing**
- Create searchable indexes using TF-IDF or BM25
- Store document embeddings or keyword indexes
- You're using `chromadb` for vector storage

### 4. **Storage**
- Save the processed chunks and indexes
- Make them ready for fast retrieval

## Your Two Ingestion Modes:

### **Full Repository** (`mode="full"`)
```python
def ingest(self):
    if self.mode == "full":
        # Process ALL files in assets/vllm-0.10.1/
        # - Python files (.py)
        # - Documentation (.md, .rst)
        # - Configuration files (.yaml, .json)
        # - Everything!
```

### **Selective Ingestion** (`mode="selective"`)
```python
def ingest(self):
    if self.mode == "selective":
        # 1. Read the questions dataset JSON file
        # 2. Extract which files are mentioned in the questions
        # 3. Only process THOSE specific files
        # Much faster for testing!
```

## Example Flow:

```python
# Your ingest function should do something like:
def ingest(self):
    print(f"🔄 Starting {self.mode} ingestion...")
    
    if self.mode == "selective":
        # Load questions to find which files to process
        files_to_process = self.extract_files_from_questions()
    else:
        # Get all files in repository
        files_to_process = self.get_all_repository_files()
    
    for file_path in files_to_process:
        # 1. Read file content
        content = self.read_file(file_path)
        
        # 2. Chunk the content (using chonkie)
        chunks = self.chunk_content(content)
        
        # 3. Create searchable index (using bm25s)
        self.index_chunks(chunks, file_path)
        
        # 4. Store in vector database (using chromadb)
        self.store_chunks(chunks, file_path)
    
    print("✅ Ingestion completed!")
```

## Why Selective is Recommended for Testing:
- **Faster**: Only processes files mentioned in your test questions
- **Focused**: Ensures you have the right content for evaluation
- **Efficient**: Good for development and debugging

