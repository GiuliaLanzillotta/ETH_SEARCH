# ETH SEARCH

This code is part of the ETH Search Next Generation project. \
Here we implement the topic modelling core as shown in the figure below to improve the ETH search bar experience. 

![overview](https://github.com/GiuliaLanzillotta/ETH_SEARCH/blob/main/ETH_search_overview.png)


## Contents 
The code is divided between the `DataImportExport`, `Experiments` and `FinalModel` directories. 
* `DataImportExport` includes the initial and final steps in the pipeline: cleaning + importing input data into the graph, and loading the inferred topics into the graph. For more details on the individual notebooks see the `DataImportExport/README`.
* `Experiments` contains experiments with models that were __not__ incorporated in the final pipeline, including baselines as well as more complex models. See the project report for further details and motivation behind these models.
* `FinalModel` provides the code associated with the final topic model and the query components.


## Usage 
Following the workflow presented in the image above, first run `DataImportExport/Data_import_final.ipynb` to extract the publications metadata. In this notebook there are further instructions on how to import this metadata, including authors, publications, departments and organisations, into Neo4j. This notebook also extracts the raw data file used to fit the topic model.  

Next, head to `FinalModel/Streaming_LDA.ipynb` to train the final model. This notebook trains two separate models (differentiated by their pre-processing) for the graph and the embedding space. These are then fed into the respective components of the query system. 

On the graph side, refer to the notebook `DataImportExport/Topics_import.ipynb`. Assuming trained streaming LDA models, it extracts the files necessary to enrich the Neo4j graph with topics. It also includes the required import queries. To further enrich the search capabilities with expert scores (see report for details), head to `FinalModel/Python_Neo4j_queries.ipynb`. Example queries can be found in `FinalModel/Graph_Queries.txt`.

As for the embeddings, this pipeline is found in `FinalModel/Embeddings_LDA.ipynb`. Assuming trained streaming LDA models, it creates the embedding representations and provides example queries & visualisations.
