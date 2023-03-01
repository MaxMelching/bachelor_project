"""
This script downloads all nocosmo files from GWTC-1, -2.1, -3 into the
path specified by `directory` (relative to where this python script is
stored). By modifying the template filename and link, it is also
possible to download other data (to see all catalogs, print
`find_datasets(type='catalog')`).

To download the cosmo files, simply change the suffixes that are used
for the variable `filename`.

To download only specific events or catalog data, simply modify
`eventlist1` etc (e.g. set them to `[]`).


Remark: the files will be downloaded into the directory where this file
is if `directory = ''`.

If there are files with the same name in the directory (for example from
previous downloads), thwill not be downloaded again.
"""

# -----------------------------------------------------------------------------
#
# This script can be used to download gravitational-wave posteriors
#
# Author: Max Melching
# Source: https://github.com/MaxMelching/bachelor_project
# 
# -----------------------------------------------------------------------------


import os
from gwosc.datasets import find_datasets


directory = '../GW_data/'



eventlist1 = ['GW150914_095045', 'GW151012_095443', 'GW151226_033853',
              'GW170104_101158', 'GW170608_020116', 'GW170729_185629',
              'GW170809_082821', 'GW170814_103043', 'GW170818_022509',
              'GW170823_131358'
              ]

for event in eventlist1:
    filename = 'IGWN-GWTC2p1-v2-' + event + '_PEDataRelease_mixed_nocosmo.h5'
    # [:-3] is to cut off '-v1'/ 'v2'/ ... at the end, which is not
    # present in the filenames and thus should be cut off
    link = 'https://zenodo.org/record/6513631/files/' + filename + '?download=1'

    try:
        # If file already exists, do not download it
        open(directory + filename, 'r').close()
    except FileNotFoundError:
        os.system('wget -O ' + directory + filename + ' ' + link)
        # The pesummary function "fetch_open_samples('eventname')" could
        # probably be used just as well (not tested)



eventlist2 = []  # In case nothing shall be downloaded from them, saves
                 # some time in script running

# Uncomment if data shall be downloaded
# eventlist2 = find_datasets(
#                            catalog='GWTC-2.1-confident',
#                            type='event',
#                            match='GW'
#                            )

for event in eventlist2:
    filename = 'IGWN-GWTC2p1-v2-' + event[:-3] + '_PEDataRelease_mixed_nocosmo.h5'
    # [:-3] is to cut off '-v1'/ 'v2'/ ... at the end, which is not
    # present in the filenames and thus should be cut off
    link = 'https://zenodo.org/record/6513631/files/' + filename + '?download=1'

    try:
        # If file already exists, do not download it
        open(directory + filename, 'r').close()
    except FileNotFoundError:
        os.system('wget -O ' + directory + filename + ' ' + link)
        # The pesummary function "fetch_open_samples('eventname')" could
        # probably be used just as well (not tested)



eventlist3 = []  # In case nothing shall be downloaded from them, saves
                 # some time in script running

# Uncomment if data shall be downloaded
# eventlist3 = find_datasets(
#                            catalog='GWTC-3-confident',
#                            type='event',
#                            match='GW'
#                            )

# eventlist3.insert(14, 'GW200105_162426')  # Add event that is not in
#                                           # confident because of
#                                           # p_astro < 0.5, but still
#                                           # considered in GWTC-3

for event in eventlist3:
    filename = 'IGWN-GWTC3p0-v1-' + event[:-3] + '_PEDataRelease_mixed_nocosmo.h5'
    # [:-3] is to cut off '-v1'/ 'v2'/ ... at the end, which is not
    # present in the filenames and thus should be cut off
    link = 'https://zenodo.org/record/5546663/files/' + filename + '?download=1'

    try:
        # If file already exists, do not download it
        open(directory + filename, 'r').close()
    except FileNotFoundError:
        os.system('wget -O ' + directory + filename + ' ' + link)
        # The pesummary function "fetch_open_samples('eventname')" could
        # probably be used just as well (not tested)