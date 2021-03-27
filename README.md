# ETH SEARCH

This code is part of the ETH Search Next Generation project. \
Here we implement the topic modelling core as shown in the figure below to improve the ETH search bar experience. 

![overview](https://github.com/GiuliaLanzillotta/ETH_SEARCH/blob/main/ETH_search_overview.png)


## Contents 
The code is divided between the `DataImportExport`, `Experiments` and `FinalModel` directories. 
* `DataImportExport` includes the initial and final steps in the pipeline: cleaning + importing input data into the graph, and loading the inferred topics into the graph. For more details on the individual notebooks see the `DataImportExport/README`.
* `Experiments`
* `FinalModel`


## Usage 
Following the workflow presented in the image above, first run `DataImportExport/DataImportFinal.ipynb` to extract the publications metadata. In this notebook there are further instructions on how to import this metadata, including authors, publications, departments and organisations, into Neo4j. This notebook also extracts the raw data file used to fit the topic model.  
