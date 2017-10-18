# Project_Veritas :bowtie:

This is project `Veritas`! It serves as an academic project for
CS5100: Foundation for Artificial Intelligence in Northeastern Univ.

## Contributors

Shubhi, Emily Dutile and Linghan Xing are the first contributors.

## What it does :rocket:

The project is an automated approach to identify authenticate news
from fake ones. 

## Approach

The project evolve three parts:

1. Parsing information

The goal of parsing is to retrieve key information for us to verity 
in the next step. Things that we are targeting at this stage is:

* External links
* Author
* References

2. Analyse texts through searching

For each identified key item we run a DFS to pull out the 
pointed sources. 

3. Evaluate and reinforce learning to improve results.
