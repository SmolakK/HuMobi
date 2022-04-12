# HuMobi
 
 ## Table of contents
* [General info](#General-info)
* [Installing HuMobi](#Installing-HuMobi)
* [Data reading](#Data-reading)
* [Data preprocessing](#Data-preprocessing)
 
## General Info
This is the HuMobi library. It is a dedicated Python library for human mobility data processing, which mostly extends Pandas
DataFrames and Geopandas GeoDataFrames to facilitate operating on a very specific data structure of individual mobility trajectories. Below you will find info how to install a HuMobi library at your computer and some demos covering most of library functionalities.

This library is mainly devoted to processing individual mobility trajectories and focuses on human mobility prediction and modelling. Initially it was implemented during PhD studies by Kamil Smolak, a PhD candidate at the Wrocław University of Environmental and Life Sciences.

If you use this library please cite below work:
Smolak, K., Siła-Nowicka, K., Delvenne, J. C., Wierzbiński, M., & Rohm, W. (2021). The impact of human mobility data scales and processing on movement predictability. Scientific Reports, 11(1), 1-10.

It is a constantly expanding project, and new functionalities are added as you read that text. Currently, I am implementing human mobility models - these are not functioning properly yet and are not covered in the documentation.

Current functionalities of HuMobi library cover:
* The basic class is TrajectoriesFrame - used to load and store the mobility data. You will find that class in the
structures directory.
* Measures contains individual and collective measures for human mobility.
* Misc contains useful functions for data processing.
* Models contains human mobility models. Currently under development.
* Predictors contains next-place prediction methods, Markov Chains, deep-learning and shallow-learning models.
* Preprocessing contains methods for data aggregation and filtering.
* Tools contain useful tools for data processing.

## Installing HuMobi
