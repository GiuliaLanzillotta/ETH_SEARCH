`DataImportExport` includes the initial and final steps in the pipeline.

* `DataImportFinal.ipynb` contains all the necessary code to open the research collection files, extract relevant information, apply preprocessing and preparing the data for the export into the graph. It also contains the Neo4j import queries. 

* `Topics_import.ipynb` covers the necessary steps to enrich the graph with the trained topic models. Note that to succesfully use this notebook you need to first run the `FinalModel/Streaming LDA` notebook and store the models on disk.

* `Data import.ipynb`, `Data exploration.ipynb` exploratory notebooks. 
