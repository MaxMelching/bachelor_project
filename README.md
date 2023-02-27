[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# bachelor_project

## Description
This repository contains my Bachelor thesis `BA_main.pdf` and code that was used to obtain the results that are discussed in it.

It deals with Gravitationa-Wave events published by the LIGO-Virgo-Kagra collaboration in the third Gravitational-Wave catalog [GWTC-3](https://arxiv.org/abs/2111.03606). More specifically, the topic are systematics in the posterior distributions of these events (to learn more about posteriors, see chapter 2 of the [thesis](BA_main.pdf) and sources within). These posteriors can be downloaded from the corresponding [Zenodo release](https://zenodo.org/record/5546663#.Yka1ky3P1D9), for example using the script `download.py`.


Just like the file names suggest, `thesis_plots.py` and `thesis_tables.py` are scripts reproducing the figures and tables. All other files contain the functions, constants etc. that are used therein.


## Project Status
The thesis has already been submitted, so the official time of writing new code is over. However, there is still work to do:

* `functions.py`: some functions have not been added yet
* `thesis_plots.py`: some plots for chapter are still missing
* `thesis_tables.py`: tables for chapter 5 are still missing
* general: improving code and documentation


## Citation
If you happen to use some of the code, please cite this repository.


<!-- ## License
 Copyright (c) 2022 Max Melching
 
 This program is free software: you can redistribute and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation (either version 3 or any later version).
 
 This program is distributed in the hope that it will be useful, but without any warranty.

 
 A copy of the GNU General Public License can be found at <http://www.gnu.org/licenses/>. -->