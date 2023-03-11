# -----------------------------------------------------------------------------
#
# This script reproduces posterior-related figures used in the thesis
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



#%% ---------- Figures for chapter 4 ----------

# Figure 4.3 (a)
event_check(
    params,
    events3,
    displayparams=latexparams,
    directory=dir
)

# Figure 4.3 (b)
event_check(
    params,
    events3,
    normed=True,
    centered=True,
    displayparams=latexparams,
    directory=dir
)

# Figure 4.3 (c)
event_check(
    params,
    events3,
    criterion='meandiff',
    threshold=0.5,
    displayparams=latexparams,
    directory=dir
)

# Figure 4.3 (d)
event_check(
    params,
    events3,
    criterion='mediandiff',
    threshold=0.5,
    displayparams=latexparams,
    directory=dir
)


# Figure 4.5 (a)
event_correlation_2D(
    ['q', 'chi_eff'],
    events3,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.5 (b)
event_correlation_2D(
    ['q', 'chi_p'],
    events3,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [0, 1]},
    directory=dir
)

# Figure 4.5 (c)
event_correlation_2D(
    ['chi_p', 'chi_eff'],
    events3,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.5 (d)
event_correlation_2D(
    ['sample_size'],
    events3,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)


# Figure 4.6 (a)
event_correlation_2D(
    ['q', 'chi_eff'],
    gooddata + baddata,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.6 (a)
event_correlation_2D(
    ['q', 'chi_eff'],
    multimod,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)


# Figure 4.7 (a)
event_correlation_2D(
    ['q', 'chi_eff'],
    events3,
    params=['chirp_mass_source'],
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.7 (b)
event_correlation_2D(
    ['q', 'chi_eff'],
    events3,
    params=['mass_ratio'],
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.7 (c)
event_correlation_2D(
    ['q', 'chi_eff'],
    events3,
    params=['chi_eff'],
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.7 (d)
event_correlation_2D(
    ['q', 'chi_eff'],
    events3,
    params=['total_mass_source'],
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.7 (e)
event_correlation_2D(
    ['q', 'chi_eff'],
    events3,
    params=['mass_1_source'],
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.7 (f)
event_correlation_2D(
    ['q', 'chi_eff'],
    events3,
    params=['mass_2_source'],
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)


# Figure 4.8 (a)
event_correlation(
    'mass_ratio',
    events3,
    params=['chirp_mass_source'],
    displayparams=latexparams,
    directory=dir
)

# Figure 4.8 (b)
event_correlation(
    'mass_ratio',
    events3,
    params=['mass_ratio'],
    displayparams=latexparams,
    directory=dir
)

# Figure 4.8 (c)
event_correlation(
    'mass_ratio',
    events3,
    params=['total_mass_source'],
    displayparams=latexparams,
    directory=dir
)

# Figure 4.8 (d)
event_correlation(
    'mass_ratio',
    events3,
    params=['chi_eff'],
    displayparams=latexparams,
    directory=dir
)

# Figure 4.8 (e)
event_correlation(
    'mass_ratio',
    events3,
    params=['mass_1_source'],
    displayparams=latexparams,
    directory=dir
)

# Figure 4.8 (f)
event_correlation(
    'mass_ratio',
    events3,
    params=['mass_2_source'],
    displayparams=latexparams,
    directory=dir
)


# Figure 4.9 (a)
event_correlation(
    'chirp_mass_source',
    events3,
    displayparams=latexparams,
    directory=dir
)

# Figure 4.9 (b)
event_correlation(
    'total_mass_source',
    events3,
    displayparams=latexparams,
    directory=dir
)

# Figure 4.9 (c)
event_correlation(
    'chirp_mass_source',
    events3,
    criterion='mediandiff',
    threshold=0.5,
    displayparams=latexparams,
    directory=dir
)

# Figure 4.9 (d)
event_correlation(
    'total_mass_source',
    events3,
    criterion='mediandiff',
    threshold=0.5,
    displayparams=latexparams,
    directory=dir
)


# Figure 4.10 (a)
event_correlation_2D(
    ['q', 'chi_eff'],
    events1comp,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.10 (b)
event_correlation_2D(
    ['q', 'chi_eff'],
    events2comp,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.10 (c)
event_correlation_2D(
    ['q', 'chi_eff'],
    events3,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)

# Figure 4.10 (d)
event_correlation_2D(
    ['q', 'chi_eff'],
    events1comp + events2comp + events3,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 1], 'y': [-1, 1]},
    directory=dir
)


# Figure 4.11 (a)
event_correlation_2D(
    ['sample_size'],
    events1comp,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 350_000], 'y': [0, 250_000]},
    directory=dir
)

# Figure 4.11 (b)
event_correlation_2D(
    ['sample_size'],
    events2comp,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 350_000], 'y': [0, 250_000]},
    directory=dir
)

# Figure 4.11 (c)
event_correlation_2D(
    ['sample_size'],
    events3,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 350_000], 'y': [0, 250_000]},
    directory=dir
)

# Figure 4.11 (d)
event_correlation_2D(
    ['sample_size'],
    events1comp + events2comp + events3,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 350_000], 'y': [0, 250_000]},
    directory=dir
)


# Figure 4.12 (a)
event_correlation_2D(
    ['chirp_mass_source'],
    events1comp,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 70], 'y': [0, 0.5]},
    directory=dir
)

# Figure 4.12 (b)
event_correlation_2D(
    ['chirp_mass_source'],
    events2comp,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 70], 'y': [0, 0.5]},
    directory=dir
)

# Figure 4.12 (c)
event_correlation_2D(
    ['chirp_mass_source'],
    events3,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 70], 'y': [0, 0.5]},
    directory=dir
)

# Figure 4.12 (d)
event_correlation_2D(
    ['chirp_mass_source'],
    events1comp + events2comp + events3,
    params=params,
    displayparams=latexparams,
    limits={'x': [0, 70], 'y': [0, 0.5]},
    directory=dir
)


# Figure 4.13 (a)
event_check(
    params,
    events1comp + events2comp,
    displayparams=latexparams,
    directory=dir
)

# Figure 4.13 (b)
event_check(
    params,
    events1comp + events2comp,
    normed=True,
    centered=True,
    displayparams=latexparams,
    directory=dir
)

# Figure 4.13 (c)
event_check(
    params,
    events1comp + events2comp,
    criterion='meandiff',
    threshold=0.5,
    displayparams=latexparams,
    directory=dir
)

# Figure 4.13 (d)
event_check(
    params,
    events1comp + events2comp,
    criterion='mediandiff',
    threshold=0.5,
    displayparams=latexparams,
    directory=dir
)


#%% ---------- Figures for chapter 5 ----------

# Figure 5.3 (a)
event_check_PCs(
    'Phenom',
    paramsmod,
    events1comp + events2comp,
    directory=dir
)

# Figure 5.3 (b)
event_check_PCs(
    'EOB',
    paramsmod,
    events1comp + events2comp,
    directory=dir
)


# Figure 5.4 (a)

compare_quality(
    paramsmod,
    events3,
    basis='Phenom',
    normed=True,
    directory=dir
)

# Figure 5.4 (b)

compare_quality(
    paramsmod,
    events3,
    basis='Phenom',
    normed=True,
    centered=True,
    directory=dir
)

# Figure 5.4 (c)

compare_quality(
    paramsmod,
    events3,
    basis='Phenom',
    normed=True,
    criterion='meandiff',
    threshold=0.5,
    directory=dir
)

# Figure 5.4 (d)

compare_quality(
    paramsmod,
    events3,
    basis='Phenom',
    normed=True,
    criterion='mediandiff',
    threshold=0.5,
    directory=dir
)


# Figure 5.5 (a)

compare_quality_points(
    paramsmod,
    baddata + gooddata,
    basis='Phenom',
    normed=True,
    directory=dir
)

# Figure 5.5 (b)

compare_quality_points(
    paramsmod,
    baddata + gooddata,
    basis='Phenom',
    normed=True,
    centered=True,
    directory=dir
)

# Figure 5.5 (c)

compare_quality_points(
    paramsmod,
    baddata + gooddata,
    basis='Phenom',
    normed=True,
    criterion='meandiff',
    threshold=0.5,
    directory=dir
)

# Figure 5.5 (d)

compare_quality_points(
    paramsmod,
    baddata + gooddata,
    basis='Phenom',
    normed=True,
    criterion='mediandiff',
    threshold=0.5,
    directory=dir
)


# Figure 5.6 (a)

covariance_avg(
    events3,
    paramsmod,
    displayparams=latexparams,
)

# Figure 5.6 (b)

covariance_avg(
    events3,
    paramsmod,
    basis='Prior',
    displayparams=latexparams,
    directory=dir
)


# Figure 5.7 (a)



# Figure 5.7 (b)