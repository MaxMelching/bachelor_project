# -----------------------------------------------------------------------------
# 
# This script reproduces posterior-related tables used in the thesis
#
# Author: Max Melching
# Source: https://github.com/MaxMelching/bachelor_project
# 
# -----------------------------------------------------------------------------



#%% ---------- Import of needed packages ----------
import numpy as np
import pandas as pd
import seaborn as sns
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt


from constants import *
from criteria import *
from functions import *



#%% ---------- Setting plot style (optional) ----------
plt.style.use("../MATPLOTLIB_RCPARAMS.sty")
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'



#%% ---------- Tables for chapter 4 ----------

critlist = [
    {'criterion': 'jsd', 'normed': False, 'centered': False},
    {'criterion': 'jsd', 'normed': True, 'centered': True},
    {'criterion': 'meandiff', 'normed': False, 'centered': False},
    {'criterion': 'mediandiff', 'normed': False, 'centered': False}
    ]

head = r'Event & JSD & JSD 2 & Mean difference & Median difference\\'
# If columns are too wide due to names of last two criteria, use column type
# "{c c c >{\centering}m{\widthof{difference}}
#  >{\centering\arraybackslash}m{\widthof{difference}} }"
# instead of "{c c c c c }", which forces a breakup into two rows


# Remark: if the tables are too wide, "\resizebox{\textwidth}{!}{% ... }" can
# be used, where the dots contain the tabular environment


# Table 4.1
generate_table(
    params,
    latexparams,
    events = events3,
    criteria = critlist,
    thresholds = [0.05, 0.05, 0.5, 0.5],  # 50% thresholds
    header = head,
    directory = dir
)

# Table 4.2
generate_table(
    params,
    latexparams,
    events = events3,
    criteria = critlist,
    thresholds = [0.05, 0.05, 0.5, 0.5],  # 50% thresholds
    header = head,
    model = 'cosmo',
    directory = dir
)

# Table 4.3
generate_table(
    params,
    latexparams,
    events = events3,
    criteria = critlist,
    thresholds = [0.01, 0.01, 0.2, 0.2],  # 20% thresholds
    header = head,
    directory = dir
)

# Table 4.4
generate_table(
    params,
    latexparams,
    events = events3,
    criteria = critlist,
    thresholds = [0.01, 0.01, 0.2, 0.2],  # 20% thresholds
    header = head,
    model = 'cosmo',
    directory = dir
)


critlist = [
    {'criterion': 'jsd', 'normed': False, 'centered': False},
    {'criterion': 'jsd', 'normed': False, 'centered': False},
    {'criterion': 'jsd', 'normed': True, 'centered': True},
    {'criterion': 'jsd', 'normed': True, 'centered': True},
    {'criterion': 'meandiff', 'normed': False, 'centered': False},
    {'criterion': 'meandiff', 'normed': False, 'centered': False},
    {'criterion': 'mediandiff', 'normed': False, 'centered': False},
    {'criterion': 'mediandiff', 'normed': False, 'centered': False}
    ]

header = r'\multirow{2}*{Eventlist} & \multirow{2}*{Parameter} &'\
    r'\multicolumn{2}{c}{JSD} & \multicolumn{2}{c}{JSD 2} & '\
    r'\multicolumn{2}{c}{Mean difference} & '\
    '\multicolumn{2}{c}{Median difference} \\\\\n'\
    r' & & 50\% & 20\% & 50\% & 20\% & 50\% & 20\% & 50\% & 20\% \\'
# Not always working, depends on number of parameters in params (multicolumn is
# the problem, so solution then is to write them into columns manually)

# Table 4.5
generate_statistics(
    params,
    latexparams,
    eventlists = [baddata, events3, gooddata],
    criteria = critlist,
    thresholds = [0.05, 0.01, 0.05, 0.01, 0.5, 0.2, 0.5, 0.2],
    header = header,
    directory = dir
    )

# Table 4.6
generate_statistics(
    params,
    latexparams,
    eventlists = [baddata, events3, gooddata],
    criteria = critlist,
    thresholds = [0.05, 0.01, 0.05, 0.01, 0.5, 0.2, 0.5, 0.2],
    header = header,
    model = 'cosmo',
    directory = dir
    )

# Table 4.7 was made by hand


# Table 4.8
generate_statistics(
    params,
    latexparams,
    eventlists = [events1comp, events2comp, events1comp + events2comp, events3],
    criteria = critlist,
    thresholds = [0.05, 0.01, 0.05, 0.01, 0.5, 0.2, 0.5, 0.2],
    header = head,
    directory = dir
    )

# Table 4.9 was made by hand


# Tabl2 4.10
generate_table(
    params,
    latexparams,
    events = events1comp + events2comp,
    criteria = critlist,
    thresholds = [0.05, 0.05, 0.5, 0.5],
    header = header,
    directory = dir
)

# Table 4.11
generate_table(
    params,
    latexparams,
    events = events1comp + events2comp,
    criteria = critlist,
    thresholds = [0.01, 0.01, 0.2, 0.2],
    header = header,
    directory = dir
)



#%% ---------- Tables for chapter 5 ----------


