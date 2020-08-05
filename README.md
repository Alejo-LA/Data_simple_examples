# BASIC TOOLS

### Description
This repository contains just a 
few helper functions for loading data with python, different cosine and jaccard implementations, 
and more tools for coding and data science that will be added along the time

### Launch
1. This repository was developed using python 3.7.7.
2. The repository provides a requirements.txt file with all the necessary dependencies. 
They can be installed with: `pip3 install -r requirements.txt`

In case you just want part of the code be check the required modules and libraries from requirements.txt

### Distances_implementations

This folder contains a few examples for some of the most common distance used in data science: 
so far cosine and Jaccard
The idea is to use this code to write entries in a Data Science Blog, so the contain can change along the time.
Some of the implementations use sparse matrices since they where done to be used in content-based recommender systems 
and collaborative factorization models, where almost all the entries are zero.
They are also differentiated based on the matrix dimension from which we take the vectors. 
Therefore, some of the implementations are just the transpose of the another.

### Loading_Data

This folder contains a few examples of how load or write file from some of the most common formats
using python. 

