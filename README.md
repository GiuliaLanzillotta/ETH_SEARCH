# ETH SEARCH

This code is part of the ETH Search Next Generation project. \
Here we implement the topic modelling core as shown in the figure below to improve the ETH search bar experience. 

![overview](https://github.com/GiuliaLanzillotta/ETH_SEARCH/blob/main/ETH_search_overview.png)


## Contents 
The code is divided between the `DataImportExport`, `Experiments` and `FinalModel` directories. 
* `DataImportExport` includes the initial and final steps in the pipeline: cleaning + importing input data into the graph, and loading the inferred topics into the graph. For more details on the individual notebooks see the `DataImportExport/README`.
* `Experiments` contains experiments with models that were __not__ incorporated in the final pipeline, including baselines as well as more complex models. See the project report for further details and motivation behind these models.
* `FinalModel`


## Usage 
Following the workflow presented in the image above, first run `DataImportExport/Data_import_final.ipynb` to extract the publications metadata. In this notebook there are further instructions on how to import this metadata, including authors, publications, departments and organisations, into Neo4j. This notebook also extracts the raw data file used to fit the topic model.  

Next, head to FinalModel/Streaming_LDA.ipynb to train the final model. This notebook trains two separate models (differentiated by their pre-processing) for the graph and the embedding space. 
