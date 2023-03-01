# -----------------------------------------------------------------------------
# 
# This script contains constants used to produce the results
#
# Author: Max Melching
# Source: https://github.com/MaxMelching/bachelor_project
# 
# -----------------------------------------------------------------------------



import h5py
from gwosc.datasets import find_datasets



#%% ---------- Constants to be set for use ----------

# Set directory where event data is stored (relative to directory of this file)
dir = '../GW_data/'

# Set cosmological model determining suffix of file names
model = 'nocosmo'



#%% ---------- Important general constants ----------

# Key set of parameters that was selected for analysis in chapter 4
params = [
    'chirp_mass_source', 'mass_ratio', 'chi_eff', 'chi_p',
    'luminosity_distance', 'theta_jn', 'total_mass_source',
    'mass_1_source','mass_2_source'
    ]

# Shorter version of params used in chapter 5
paramsmod = [
    'chirp_mass_source', 'mass_ratio', 'chi_eff', 'chi_p', 
    'luminosity_distance', 'theta_jn'
    ]

# Dictionary with parameters as keys and display versions of parameters
# as values (e.g. LaTeX code for them in a math environment)
latexparams = {
    'chirp_mass_source': r'$\mathcal{M}$', 'chirp_mass': r'$\mathcal{M}$',
    'mass_ratio': '$q$', 'total_mass_source': '$M$', 'mass_1_source': '$m_1$',
    'mass_1': '$m_1$', 'mass_2_source': '$m_2$', 'mass_2': '$m_2$',
    'chi_eff': '$\chi_{eff}$', 'chi_p': r'$\chi_p$', 'a_1': '$\chi_1$',
    'a_2': '$\chi_2$', 'luminosity_distance': '$D_L$',
    'theta_jn': r'$\theta_{jn}$', 'iota': r'$\iota$',
    'log_likelihood': '$\log\mathcal{L}$',
    'network_optimal_snr': r'$\rho_{opt}$ (network)',
    'network_matched_filter_snr': r'$\rho_{matched filter}$ (network)'
    }


# Events from GWTC-1 (manual typing necessary because find_datasets only gives
# names like GW150914, but files from Zenodo are named differently)
events1 = [
    'GW150914_095045', 'GW151012_095443', 'GW151226_033853', 'GW170104_101158',
    'GW170608_020116', 'GW170729_185629', 'GW170809_082821', 'GW170814_103043',
    'GW170818_022509', 'GW170823_131358'
    ]


# Events from GWTC-2.1 (without those who are already in GWTC-1)
events2 = find_datasets(
    catalog='GWTC-2.1-confident',
    type='event',
    match='GW'
    )

# Discard version at the end of names
events2 = [event[:-3] for event in events2]


# Events from GWTC-3
events3 = find_datasets(
    catalog='GWTC-3-confident',
    type='event',
    match='GW'
    )

# Discard version at the end of names
events3 = [event[:-3] for event in events3]

# Add event GW200105_162426 as it is considered in GWTC-3, but not in the
# confident events as it has p_astro < 0.5
events3.insert(14, 'GW200105_162426')


# Create list where all GW events are contained
allevents = events1 + events2 + events3


# Events from GWTC-3 marked for high disagreement by criteria used
baddata = [
    'GW191109_010717', 'GW191127_050227', 'GW191216_213338', 'GW191219_163120',
    'GW200112_155838', 'GW200129_065458', 'GW200208_222617', 'GW200308_173609',
    'GW200316_215756', 'GW200322_091133'
    ]
# Manually excluded were: 'GW200105_162426', 'GW200210_092254'

# Events from GWTC-3 marked for low disagreement by criteria used
gooddata = [
    'GW191126_115259', 'GW191215_223052', 'GW191222_033537', 'GW200208_130117',
    'GW200209_085452', 'GW200216_220804', 'GW200220_124850', 'GW200306_093714'
    ]

# Events mentioned in GWTC-3 to have multimodality
multimod = [
    'GW200208_222617', 'GW200308_173609', 'GW200322_091133', 'GW200129_065458',
    'GW200225_060421', 'GW200306_093714'
    ]

# Events from GWTC-1, -2.1 marked for high disagreement by criteria used
# (but not checked as carefully as baddata, gooddata)
baddatacomp = [
    'GW150914_095045', 'GW170814_103043', 'GW190412_053044', 'GW190517_055101',
    'GW190519_153544', 'GW190521_030229', 'GW190521_074359', 'GW190527_092055',
    'GW190620_030421', 'GW190707_093326', 'GW190708_232457', 'GW190720_000836',
    'GW190924_021846', 'GW190930_133541'
    ]

# Events which do not have particularly bad or good agreement
normaldata = []
for event in events3:
    if event not in gooddata and event not in baddata:
        normaldata += [event]



#%% ---------- Generate lists with special events ----------

# Extract events with samples for both waveform models we use from events1
events1comp = [] 
for event in events1:
    filename = 'IGWN-GWTC2p1-v2-' + event + '_PEDataRelease_mixed_' + model + '.h5'
    file = h5py.File(dir + filename, 'r')

    if 'C01:IMRPhenomXPHM' in file.keys() and 'C01:SEOBNRv4PHM' in file.keys():
        events1comp += [event]


# Extract events with samples for both waveform models we use from events2
events2comp = []
for event in events2:
    filename = 'IGWN-GWTC2p1-v2-' + event + '_PEDataRelease_mixed_' + model + '.h5'
    file = h5py.File(dir + filename, 'r')

    if 'C01:IMRPhenomXPHM' in file.keys() and 'C01:SEOBNRv4PHM' in file.keys():
        events2comp += [event]


#  Create list where all GW events relevant for analyses are contained
alleventscomp = events1comp + events2comp + events3