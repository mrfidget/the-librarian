# The Librarian

A local, AI-powered document management system to retrive, classify, store and query different formats of documents.

## Architecture

``` mermaid
mindmap
    root((The Librarian))
            ((Downloader))
            ((Extractor))
                ((zip files))
            ((Classifier))
                ((Type of files)) 
                    ((Text))
                    ((Image))
                    ((PDF)
                    ((unknown))    
            ((Processor))
                ((Text))
                ((Image))
                ((PDF))
            ((Indexer))
                ((Vector indexer))
            ((Retriever))
                ((Query the Librarian via CLI))
```
