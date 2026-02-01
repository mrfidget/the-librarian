# The Librarian

A local, AI-powered document management system to retrive, classify, store and query different formats of documents.

## Architecture

``` mermaid
mindmap
    root(The Librarian)
        Tools
            Downloader
            Extractor
                zip files
            Classifier
                Type of file TEXT, IMAGE, PDF, or UNKNOWN    
            Processor
                Text
                Image
                PDF
            Indexer
                Vector indexer using sentence-transformers
            Retriever
                Query the Librarian via CLI
```
