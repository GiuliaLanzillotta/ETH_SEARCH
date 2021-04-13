
`FinalModel`contains the files related to the final topic model - Streaming LDA

* `Streaming_LDA.ipynb` pre-processes the data, compares the best LDA models by number of topics returned by a grid search in the `Experiments` directory and trains the final model. It assumes that the user has already retrieved `abstracts_eng.csv`. It exports pre-processed data files used for the graph and the embedding side, `collection_cleaned.p` and `collection_cleaned_fullwords.p` respectively, as well as the Streaming LDA models trained on four different batches, `LDA1batch1.bin`, `LDA1batch2.bin`, `LDA1batch3.bin` and `LDA1batch4.bin`.
* `Python_Neo4j_queries.ipynb` explores the mapping from ETH Search query input to Neo4j queries and, more importantly, provides the necessary Cypher code to enrich the graph with expert scores.
* `Embedding LDA.ipynb`....
