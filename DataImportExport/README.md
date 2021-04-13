`DataImportExport` includes the initial and final steps in the pipeline.

* `Data_import_final.ipynb` contains all the necessary code to open the research collection files, extract relevant information, apply preprocessing and preparing the data for the export into the graph. It also contains the Cypher queries for importing the data into Neo4j. Running this notebook extracts two csv files: `graph_data_final.csv` which contains metadata to be imported to the Neo4j database and `abstracts_eng.csv` which is used for topic modelling. 

* `Topics_import.ipynb` covers the steps to enrich the graph with the trained topic models. It exports the files `abstract+topic.csv`, `topics.csv` and `words.csv` for this purpose and provides the necessary import queries. Note that to succesfully use this notebook you need to first run the `FinalModel/Streaming LDA` notebook and store the models `LDA1batch1.bin`, `LDA1batch2.bin`, `LDA1batch3.bin` and `LDA1batch4.bin` on disk.

* `Data exploration.ipynb` is an exploratory notebook. 
