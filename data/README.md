# Data

This directory contains the main dataset to reproduce the experimental result.

As indicated in the code, the main dataset, containing the single Wikipedia articles and drembank reports used in the experiment, is 


`Wiki_no_title_reduced_DreamBank_reduced_en.csv`

To speed up the analysis, we have already included the perplexity scores produced by the adopted LLMs.

Please note that texts, year, and gender information for each series have been removed. The full text for Wikipedia articles and Dreabank reports, as well as infor for series, gender and year, can be found [here](https://github.com/lorenzoscottb/dream_perplexity/blob/main/DreamBank_en_pptx_GPT2.csv), by matching the `File_index` and `Unnamed: 0` columns of the respective csv files. More info on each DreamBank series can be found in thie [csv file](https://github.com/josauder/dreambank_visualized/blob/master/info.csv), or this [web page](https://dreambank.net/grid.cgi#b-baseline).
