# RennoCosta_Tort_eLIFE_2016

Source Files and Instructions for replication of the paper RennoCosta and Tort submited to eLIFE. This repository include the model scripts used to produce the data for the paper; the simulation scripts with the calls of the model scripts used to produce the dataset or a limited dataset, not included on the paper, but that can be produced with little computational resources; and the analysis scripts used to produce the figures;

The limited dataset is built from a limited number of simulations that should be feasible to produce in a regular desktop (expect about 2 days of computation in a 4-core laptop). Full dataset was produced with the use of a supercomputer. Scripts are made available for reference.


List of files

1) model scripts


elife_script_convergence.py       : model script for the simulations of the first theta cycle

elife_script_morph.py             : model script with whole exploration of the envrionment

elife_script_morph_noise.py       : model script, similar to previous, but with different input noise values

elife_script_morph_consistency.py : model script, similar to previous, but with different input consistency values


2) simulation scripts

elife_input_01.txt      :   simluation script for figure 3 and 4 (limited dataset)


3) analysis scripts


a. limited dataset ( files under preparation : due 20 Sep 2016 )


elife_lfigureplot_3.py    : script to plot figure 3 

elife_lfigureplot_4.py     : script to plot figure 4

elife_lfigureplot_4s2.py   : script to plot figure 4 sup 2

elife_lfigureplot_5.py     : script to plot figure 5 and 5 sup 2 to 9

elife_lfigureplot_6.py    : script to plot figure 6 and 6 sup 1




b. full dataset ( files under preparation : due 20 Sep 2016 )


elife_figureplot_3.py     : script to plot figure 3

elife_figureplot_3s1.py   : script to plot figure 3 sup 1

elife_figureplot_4.py     : script to plot figure 4

elife_figureplot_4s1.py   : script to plot figure 4 sup 1

elife_figureplot_4s2.py   : script to plot figure 4 sup 2

elife_figureplot_5.py     : script to plot figure 5 and 5 sup 2 to 9

elife_figureplot_5s1.py   : script to plot figure 5 sup 1

elife_figureplot_6.py    : script to plot figure 6 and 6 sup 1


4) support scripts

process_cluster_elife.py   : sample script used to run a simulation scripts

support_filename.py        : file position definition file


