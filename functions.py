# -----------------------------------------------------------------------------
# 
# This script contains functions and constants used to produce the results
#
# Author: Max Melching
# Source: https://github.com/MaxMelching/bachelor_project
# 
# -----------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import h5py
from gwosc.datasets import find_datasets

from pca_class import pca, create_datamatrix
from criteria import *



def event_check(params: list, events: list, criterion: str = 'jsd',
                normed: bool = False, centered: bool = False,
                model: str = 'nocosmo', plot: bool = True,
                threshold: float = 0.05, displayparams: dict = None,
                directory: str = '', exclude: list = None, limits = None,
                save: bool = False, path: str = ''):
    """
    Computes the difference between IMRPhenomXPHM and SEOBNRv4PHM 1D
    posteriors for multiple GW events according to a selected
    criterion. This is either returned or visualized.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found e.g.
          using the find_datasets function from the gwosc.datasets
          package.
        - criterion (str, optional, default = 'jsd'): specifies the
          criterion used to measure the differences. Can have values
          'jsd' (for Jensen-Shannon divergence), 'meandiff' (for mean
          difference in units of the average standard deviation) or
          'mediandiff' (for median difference in units of the average
          credible interval).
        - normed (boolean, optional, default = False): specifies if
          posteriors are normalized to have a standard deviation of 1.
        - centered (boolean, optional, default = False): specifies if
          posteriors are normalized to have a mean of 0.
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - plot (boolean, optional, default = True): specifies if output
          is a plot or the list of results.
        - threshold (float, optional, default = 0.05): value used to
          draw a line in the plot, corresponding to the crossing
          between criterion passed/ not passed. Default corresponds to
          JSD of 50% mean shifted normal distributions.
        - displayparams (dictionary, optional, default = None):
          contains parameters from params as keys and display names for
          plots as values (e.g. LaTeX code). If None, the elements from
          params are taken.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors
          are stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from events.
        - limits (tuple or list with two elements, optional,
          default = None): sets specific limits on y-axis.
        - save (boolean, optional, default = False): if True, the plot
          gets saved to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'pictures/photo.png', it is assumed
          to also contain a file name, so in the example a png-file
          named 'photo' will be saved to the subfolder 'pictures'.

    Returns:
        - Either a list containing a list with differences in the
          parameter posteriors for every GW event in events or None.
    """


    # Check conditions
    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')

    if exclude:
        for event in exclude:
            events.remove(event)
            
            
    # Store frequently used constants
    n = len(events)
    parlen = len(params)
    
    # Create list to append results from events
    critvals = []
    

    for event in events:
        # Open file containing event data
        try:
            filename = 'IGWN-GWTC3p0-v1-' + event + '_PEDataRelease_mixed_'\
                       + model + '.h5'
            file = h5py.File(directory + filename, 'r')
        except OSError:
            try:
                filename = 'IGWN-GWTC2p1-v2-' + event\
                           + '_PEDataRelease_mixed_' + model + '.h5'
                file = h5py.File(directory + filename, 'r')
            except OSError:
                print(f'No file found for event \'{event}\' in directory'\
                    f' \'{directory}\'.')

        # Extract relevant data from file
        try:
            Xph = create_datamatrix(
                file['C01:IMRPhenomXPHM/posterior_samples'], 
                params
                )
        except KeyError:
            Xph = create_datamatrix(
                file['C01:IMRPhenomXPHM:HighSpin/posterior_samples'], 
                params
                )
            
        Xse = create_datamatrix(
            file['C01:SEOBNRv4PHM/posterior_samples'], 
            params
            )


        # Apply optional corrections to data
        if centered:
            Xph -= np.mean(Xph, axis = 0)
            Xse -= np.mean(Xse, axis = 0)
        
        if normed:
            Xph /= np.std(Xph, axis = 0)
            Xse /= np.std(Xse, axis = 0)

            # If normalization is on, one has to be careful with order
            # Xse /= np.std(Xph, axis = 0)
            # Xph /= np.std(Xph, axis = 0)

            # Xph /= np.std(Xse, axis = 0)
            # Xse /= np.std(Xse, axis = 0)

            # Decided against doing this because criteria are invariant under
            # this kind of normalization, which means we can directly used
            # non-normalized data (saves operations)

        
        # Calculate value of chosen criterion for all parameters, current event
        if criterion == 'jsd':
            # Jensen-Shannon divergence
            critvals += [[jsd(Xph[:, j], Xse[:, j]) for j in range(parlen)]]

            # Title of plot which is generated in some cases
            # title = 'JSD for original parameters'

        elif criterion == 'meandiff':
            # mean difference in units of average standard deviation
            critvals += [mean_criterion(Xph, Xse)]

            # Title of plot which is generated in some cases
            # title = 'Mean difference/average stddev for original parameters'

        elif criterion == 'mediandiff':
            # median difference in units of average credible interval
            critvals += [median_criterion(Xph, Xse, 34.135)]

            # Title of plot which is generated in some cases
            # title = 'Median difference/average credible interval'\
            # 'for original parameters'

        else:
            raise ValueError('Invalid criterion.')


    # Return list with criterion values or plot of them
    if plot:
        if displayparams is None:
            displayparams = {param: param for param in params}

        fig, ax = plt.subplots(figsize = (n / 2, 6))

        # Create numpy-array for convenient access to columns
        critvals = np.array(critvals)

        x = [i for i in range(n)]

        for k in range(parlen):
            ax.plot(x, critvals[:, k], 'o', label = displayparams[params[k]])
        
        
        ax.axline((0, threshold), (n - 1, threshold), color = 'r')

        ax.legend(bbox_to_anchor = (1.0, 0.5), loc = 'center left')
        # ax.set_title(title)
        ax.grid(axis = 'y')
        
        # Set number of ticks for x-axis and label ticks
        ax.set_xticks(x, events, rotation = 45, horizontalalignment = 'right',
            rotation_mode = 'anchor')#rotation = 'vertical')

        if limits:
            ax.set_ylim(limits[0], limits[1])
        
        # Create pdf of plot
        if save:
            # If path already contains a name + data format, this name is
            # taken; otherwise, an automatic one is created
            if '.' in path[-4:]:
                plt.savefig(path, bbox_inches = 'tight')
            else:
                name = f'eventplot_{model}_crit{criterion}_{str(params)}'
                # if limits:
                #     name += f'_limits{limits}'

                if normed:
                    name += '_normed'
                if centered:
                    name += 'centered'
                
                plt.savefig(path + name + '.pdf', bbox_inches = 'tight')

        plt.show()

    else:
        return critvals


def generate_table(params: list, latexparams: dict, events: list,
                   criteria: dict, thresholds: list,
                   exceedvals: dict = {'red': 2, 'yellow': 1},
                   badevents: list = None, goodevents: list = None,
                   model: str = 'nocosmo', header: str = None,
                   caption: str = None, label: str = None, directory: str = '',
                   exclude: list = None, save: bool = False, path: str = ''):
    """
    Generates LaTeX code for a table which contains information on
    differences between IMRPhenomXPHM and SEOBNRv4PHM 1D posteriors for
    multiple GW events according to selected criteria. This table is
    either printed or saved to a text file.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - latexparams (dictionary, optional, default = None): contains
          parameters from params as keys and LaTeX code for their names
          as values. If None, the elements from params are taken.
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found
          e.g. using the find_datasets function from the gwosc.datasets
          package.
        - criteria (dict): specifies the criteria making up columns of
          the table. Has to have keys 'criterion' (string), 'normed'
          (boolean), 'centered' (boolean) where 'criterion' can have
          values allowed as criterion in event_check. These are 'jsd'
          (for Jensen-Shannon divergence), 'meandiff' (for mean
          difference in units of the average standard deviation) or
          'mediandiff' (for median difference in units of the average
          credible interval).
        - thresholds (list or array-like): contains values used to
          determine if the criterion at the corresponding index is
          passed/ not passed.
        - exceedvals (dict, optional, default = {'red': 2,
          'yellow': 1}): specifies how much parameters are allowed to
          exceed criteria before modest (yellow) or high (red)
          significance is marked. Must contain keys 'red' and 'yellow'
          with the corresponding keys being integers (how many
          parameter posteriors are allowed to violate).
        - badevents (list or array-like, optional, default = None):
          enables highlighting of events with known bad agreement in
          red (or any other color encoding). If None, no events are
          highlighted.
        - goodevents (list or array-like, optional, default = None):
          enables highlighting of events with known good agreement in
          green (or any other color encoding). If None, no events are
          highlighted.
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - header (string, optional, default = None): text to put in
          header of the table. If None, an automatic one is created.
        - caption (string, optional, default = None): text to put in
          caption of the table. If None, no caption.
        - label (string, optional, default = None): text to put in
          label of the table. If None, no label.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from events.
        - save (boolean, optional, default = False): if True, the table
          gets saved as a text file to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'results/table.txt', it is assumed
          to also contain a file name, so in the example a txt-file
          named 'table' will be saved to the subfolder 'results'.

    Returns:
        - None
    """
    
    
    # Check conditions
    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')

    if exclude:
        for event in exclude:
            events.remove(event)


    # Calculate how events perform under chosen criteria
    data = [event_check(params, events, criterion = crit['criterion'],
                        normed = crit['normed'], centered = crit['centered'],
                        model = model, plot = False, directory = directory)
            for crit in criteria]

    # Convert to numpy-arrays for convenient commands
    data = np.array(data)
    params = np.array(params)
    
    # Store frequently used constants
    critlen = len(criteria)


    # Start string which will form table in the end
    tablestr = '\\begin{table}\n\centering\n\small\n\n\\begin{tabular}{'\
               + (critlen + 1) * 'c ' + '}'
    # The command \small is necessary because table gets too wide otherwise 
    # (even smaller is \footnotesize, \resizebox also convenient).


    # Adds header to table (automatically of None is given)
    if header:
        tablestr += '\n' + header
    else:
        header = '\n'
        for crit in criteria:
            header += ' & Criterion ' + str(crit)
        header += '\\\\'
        
        tablestr += header


    for i, event in enumerate(events):
        # Add row separation and entry for first column
        if (badevents is not None) and (event in badevents):
            tablestr += '\n\midrule\n\cellcolor{red!36} ' + event[:8] + '\_'\
                        + event[9:] + ' ' # only _ produces LaTeX error
        elif (goodevents is not None) and (event in goodevents):
            tablestr += '\n\midrule\n\cellcolor{green!36} ' + event[:8] + '\_'\
                        + event[9:] + ' '
        else:
            tablestr += '\n\midrule\n' + event[:8] + '\_' + event[9:] + ' '
        # Instead of '\midrule', '\hline' could be chosen and the same is true
        # for r'\_' and r'\textunderscore', r'\verb|_|'


        # Alternative: use different cellcolors for events instead of rules
        # if i % 2 == 0:
        #     tablestr += '\n\n' + event[:8] + '\_' + event[9:] 
        # else:
        #     tablestr += '\n\n\cellcolor{gray!36} ' + event[:8] + '\_'\
        #                 + event[9:]


        # Add results for event to this row
        for j in range(critlen):
            tablestr += '& '

            # Determine number of parameters for which threshold is exceeded
            numb = len(data[j][i][data[j][i] >= thresholds[j]])

            # Based on numb, set cellcolor and add parameters which failed
            if numb >= exceedvals['red']:
                tablestr += '\cellcolor{red!36} '

                for param in params[data[j][i] >= thresholds[j]]:
                    tablestr += latexparams[param] + ' '
            elif numb >= exceedvals['yellow']:
                tablestr += '\cellcolor{yellow!36} '

                for param in params[data[j][i] >= thresholds[j]]:
                    tablestr += latexparams[param] + ' '
            else:
                tablestr += '\cellcolor{green!36} '

        # New row for new event
        tablestr += '\\\\'


    # Add (possibly empty) caption and label
    tablestr += '\n\end{tabular}\n\n\caption{'
    
    if caption:
        tablestr += caption
    
    tablestr += '}\n\label{tab:'
    
    if label:
        tablestr += label
    
    tablestr += '}\n\end{table}'


    # Create txt file containing the table or print it (e.g. for copy + paste)
    if save:
        # If path already contains a name + data format, this name is
        # taken; otherwise, an automatic one is created
        if '.' in path[-4:]:
            with open(path, 'w') as f:
                f.write(tablestr)
        else:
            # Create informative name
            txtname = path + 'eventdata_'

            for crit in criteria:
                txtname += str(crit['criterion'])
                if crit['normed']:
                    txtname += 'normed'
                if crit['centered']:
                    txtname += 'centered'
            
            # Create file
            with open(path + txtname + '.txt', 'w') as f:
                f.write(tablestr)
    else:
        print(tablestr)


def event_correlation(axis: str, events: list, params: list = None,
                      criterion: str = 'jsd', normed: bool = False,
                      centered: bool = False, model: str = 'nocosmo',
                      plot = True, threshold: float = 0.05,
                      displayparams: dict = None, limits: dict = None,
                      directory: str = '', exclude: list = None,
                      save: bool = False, path: str= ''):
    """
    Shows maximum difference between IMRPhenomXPHM and SEOBNRv4PHM 1D
    posteriors of certain parameters for multiple GW events according
    to a selected criterion. This is visualized as a colorized point
    in a 2D projection of the parameter space.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - axis (string): name of parameter whose median determines
          position of points on x-axis. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior (for this function,
          'network_optimal_snr' and 'network_matched_filter_snr' are
          also supported as arguments for axis).
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found
          e.g. using the find_datasets function from the gwosc.datasets
          package.
        - params (list or array-like, optional, default = None):
          contains names of parameters to take 1D posteriors from,
          which are used to compute the differences of which maximum
          is taken (can be different from axis). Possible names can be
          found by looking at '.dtype.names' of the posterior_samples
          attribute for the Mixed posterior. If None, only the
          parameter given as argument for 'axis' is taken.
        - criterion (str, optional, default = 'jsd'): specifies the
          criterion used to measure the differences. Can have values
          'jsd' (for Jensen-Shannon divergence), 'meandiff' (for mean
          difference in units of the average standard deviation) or
          'mediandiff' (for median difference in units of the average
          credible interval).
        - normed (boolean, optional, default = False): specifies if
          posteriors are normalized to have a standard deviation of 1.
        - centered (boolean, optional, default = False): specifies if
          posteriors are normalized to have a mean of 0.
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - plot (boolean, optional, default = True): specifies if output
          is a plot or a list with list of results  (first element is
          list with x-values, second contains y-values).
        - threshold (float, optional, default = 0.05): value used to
          draw a line in the plot, corresponding to the crossing
          between criterion passed/ not passed. Default corresponds to
          JSD of 50% mean shifted normal distributions.
        - displayparams (dictionary, optional, default = None):
          contains parameters from params as keys and display names for
          plots as values. If None, the elements from params are taken.
        - limits (dict, optional, default = None): allows custom limits
          in the plot. Has to have keys 'x' and/ or 'y' with each value
          being a list/ tuple/ array-like. The first two elements of
          each value are taken as lower, upper limit (in case they
          exist; otherwise, the limits are choosen automatically by
          matplotlib).
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from eventlist.
        - save (boolean, optional, default = False): if True, the plot
          gets saved to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'pictures/photo.png', it is assumed
          to also contain a file name, so in the example a png-file
          named 'photo' will be saved to the subfolder 'pictures'.

    Returns:
        - None or the arrays for x-, y-axis (depending on value of plot).
    """

    
    # Check conditions
    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')

    if params is None:
        params = [axis]
    
    if (('network_optimal_snr' in params)
        or ('network_matched_filter_snr' in params)):
        raise ValueError('\'network_optimal_snr\' and'\
            '\'network_matched_filter_snr\' are not allowed as parameters, '\
            'only as axis.')
    
    if exclude:
        for event in exclude:
            events.remove(event)
            
    
    # Store frequently used constants
    parlen = len(params)

    
    # Create lists to append results from events
    x = []  # values for x-axis
    critvals = []  # values for y-axis
    
    for event in events:
        # Open file containing event data
        try:
            filename = 'IGWN-GWTC3p0-v1-' + event + '_PEDataRelease_mixed_'\
                       + model + '.h5'
            file = h5py.File(directory + filename, 'r')
        except OSError:
            try:
                filename = 'IGWN-GWTC2p1-v2-' + event\
                           + '_PEDataRelease_mixed_' + model + '.h5'
                file = h5py.File(directory + filename, 'r')
            except OSError:
                print(f'No file found for event \'{event}\' in directory'\
                    f' \'{directory}\'.')

        # Extract relevant data from file
        try:
            Xph = create_datamatrix(
                file['C01:IMRPhenomXPHM/posterior_samples'], 
                params
                )
        except KeyError:
            Xph = create_datamatrix(
                file['C01:IMRPhenomXPHM:HighSpin/posterior_samples'], 
                params
                )
        
        Xse = create_datamatrix(
            file['C01:SEOBNRv4PHM/posterior_samples'], 
            params
            )


        # Apply optional corrections to data
        if centered:
            Xph -= np.mean(Xph, axis = 0)
            Xse -= np.mean(Xse, axis = 0)
        
        if normed:
            Xph /= np.std(Xph, axis = 0)
            Xse /= np.std(Xse, axis = 0)


        # Compute medians; special case are SNR values, which are only
        # stored in Phenom model, not EOB or Mixed
        try:
            x += [np.percentile(file['C01:Mixed/posterior_samples'][axis], 50)]
        except ValueError:
            if ((axis == 'network_optimal_snr')
                or (axis == 'network_matched_filter_snr')):
                try:
                    x += [np.percentile(
                        file['C01:IMRPhenomXPHM/posterior_samples'][axis], 
                        50
                        )]
                except KeyError:
                    x += [np.percentile(
                        file['C01:IMRPhenomXPHM:HighSpin/posterior_samples'
                            ][axis], 
                        50
                        )]
            elif axis == 'prior':
                Xmixed = create_datamatrix(
                    file['C01:Mixed/posterior_samples'], 
                    params
                    )
                
                try:
                    prior = create_datamatrix(
                        file['C01:IMRPhenomXPHM/priors/samples'], 
                        params
                        )
                except KeyError:
                    try:
                        prior = create_datamatrix(
                            file['C01:IMRPhenomXPHM:HighSpin/priors/samples'], 
                            params
                            )
                    except KeyError:
                        # print(f'{event} has no prior samples')
                        continue # no prior samples for an event -> skip it
                
                
                # Apply optional corrections to data
                if centered:
                    Xmixed -= np.mean(Xmixed, axis = 0)
                    prior -= np.mean(prior, axis = 0)
                
                if normed:
                    Xmixed /= np.std(Xmixed, axis = 0)
                    prior /= np.std(prior, axis = 0)


                # x += [np.amax([kld(Xmixed[:, j], prior[:, j])
                #                for j in range(parlen)])]
                # x += [np.amax([kld(prior[:, j], Xmixed[:, j])
                #                for j in range(parlen)])]

                x += [np.amax([jsd(Xmixed[:, j], prior[:, j])
                      for j in range(parlen)])]
            else:
                raise ValueError('Invalid input for axis.')

        
        # Calculate value of chosen criterion for all parameters, current event
        if criterion == 'jsd':
            # Jensen-Shannon divergence
            critvals += [np.amax([jsd(Xph[:, j], Xse[:, j])
                         for j in range(parlen)])]

            # Label for y-axis
            crit = 'Maximum JSD'

        elif criterion == 'meandiff':
            # Mean difference in units of average standard deviation            
            critvals += [np.amax(mean_criterion(Xph, Xse))]

            # Label for y-axis
            crit = 'Maximum mean difference/average stddev'

        elif criterion == 'mediandiff':
            # Median difference in units of average credible interval
            critvals += [np.amax(median_criterion(Xph, Xse, 34.135))]

            # Label for y-axis
            crit ='Maximum median difference/average credible interval'

        else:
            raise ValueError('Invalid criterion.')


    if plot:
        fig, ax = plt.subplots(figsize = (6, 6))
        
        x, critvals = np.array(x), np.array(critvals)
        ax.plot(x[critvals <= threshold], critvals[critvals <= threshold],
                'o', color = 'g', label = 'threshold passed')
        ax.plot(x[critvals > threshold], critvals[critvals > threshold],
                'o', color = 'r', label = 'threshold failed')

        # KLD can be infinity, which cannot be displayed in plot, and we fix
        # that by plotting them separately with a certain finite value
        if (axis == 'prior') and (np.Inf in x):
            ax.plot(
                np.array(len(x) * [1.1 * np.amax(x[x < np.Inf])])[x == np.Inf],
                critvals[x == np.Inf], 
                'o', 
                color = 'b', 
                label = 'KLD = $\infty$'
                )

        if displayparams is None:
            latexparam = axis
        else:
            latexparam = displayparams[axis]
        
        # ax.set_title(f'Correlation of {latexparam} and waveform agreement')
        ax.legend()
        ax.grid(True)
        ax.set_ylabel(crit)
        ax.set_xlabel(latexparam)

        if limits is not None:
            # It is possible that only x- or y-limits are given,
            # so accessing the other ones would cause an error without try
            try:
                ax.set_xlim(limits['x'][0], limits['x'][1])
            except KeyError:
                pass

            try:
                ax.set_ylim(limits['y'][0], limits['y'][1])
            except KeyError:
                pass
            
        
        # Create pdf of plot
        if save:
            # If path already contains a name + data format, this name is
            # taken; otherwise, an automatic one is created
            if '.' in path[-4:]:
                plt.savefig(path, bbox_inches = 'tight')
            else:
                name = f'event_correlation_{model}_crit{criterion}_{axis}'
                # if limits:
                #     name += f'_limits{limits}'
                # gives error due to ':'

                if params is not None:
                    name += '_' + str(params)
                if normed:
                    name += '_normed'
                if centered:
                    name += 'centered'
                
                plt.savefig(path + name + '.pdf', bbox_inches = 'tight')

        plt.show()
    else:
        return [x, critvals]


def event_correlation_2D(axes: list, events: list, params: list = None,
                         criterion: str = 'jsd', normed: bool = False,
                         centered: bool = False, model: str = 'nocosmo',
                         threshold: float = 0.05, displayparams: dict = None,
                         limits: dict = None, palette: str = 'flare',
                         directory: str = '', exclude: list = None,
                         save: bool = False, path: str= ''):
    """
    Shows maximum difference between IMRPhenomXPHM and SEOBNRv4PHM 1D
    posteriors of certain parameters for multiple GW events according
    to a selected criterion. This is visualized as a colorized point in
    a 2D projection of the parameter space.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - axes (list or array-like): contains names of parameters whose
          medians determine position of points in the respective plane.
          Must have length 2 (exception: 'axes = ['sample_size']').
          Possible names can be found by looking at '.dtype.names' of
          the posterior_samples attribute for the Mixed posterior (for
          this function, '['sample_size']' is also supported as an
          argument for axes).
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found
          e.g. using the find_datasets function from the gwosc.datasets
          package.
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from, which are used to compute the
          differences of which maximum is taken (can be different from
          axes).  Possible names can be found by looking at
          '.dtype.names' of the posterior_samples attribute for the
          Mixed posterior. If None, the parameters given as arguments
          in 'axes' are taken.
        - criterion (str, optional, default = 'jsd'): specifies the
          criterion used to measure the differences. Can have values
          'jsd' (for Jensen-Shannon divergence), 'meandiff' (for mean
          difference in units of the average standard deviation) or
          'mediandiff' (for median difference in units of the average
          credible interval).
        - normed (boolean, optional, default = False): specifies if
          posteriors are normalized to have a standard deviation of 1.
        - centered (boolean, optional, default = False): specifies if
          posteriors are normalized to have a mean of 0.
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - threshold (float, optional, default = 0.05): value used to
          draw a line in the plot, corresponding to the crossing
          between criterion passed/ not passed. Default corresponds to
          JSD of 50% mean shifted normal distributions.
        - displayparams (dictionary, optional, default = None):
          contains parameters from params as keys and display names for
          plots as values. If None, the elements from params are taken.
        - limits (dict, optional, default = None): allows custom limits
          in the plot. Has to have keys 'x' and/ or 'y' with each value
          being a list/ tuple/ array-like. The first two elements of
          each value are taken as lower, upper limit (in case they
          exist; otherwise, the limits are choosen automatically by
          matplotlib).
        - palette (string, optional, default = 'flare'): color palette
          chosen to colorize points for median values based on the
          value of criterion chosen to compute difference of
          corresponding posteriors.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from eventlist.
        - save (boolean, optional, default = False): if True, the plot
          gets saved to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'pictures/photo.png', it is assumed
          to also contain a file name, so in the example a png-file
          named 'photo' will be saved to the subfolder 'pictures'.

    Returns:
        - None
    """

    
    # Check conditions
    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')
    
    if (len(axes) != 2) and ('sample_size' not in axes):
        raise ValueError('Axes must have length 2 or contain \'sample_size\'.')
    
    if params is None:
        params = axes
    
    if exclude:
        for event in exclude:
            events.remove(event)
            
    
    # Store frequently used constants
    parlen = len(params)

    
    # create lists to append results from events
    x = []  # Values for x-axis
    y = []  # Values for y-axis
    critvals = [] # Values used for coloring, marker etc
    
    for event in events:
        # open file containing event data
        try:
            filename = 'IGWN-GWTC3p0-v1-' + event + '_PEDataRelease_mixed_'\
                       + model + '.h5'
            file = h5py.File(directory + filename, 'r')
        except OSError:
            try:
                filename = 'IGWN-GWTC2p1-v2-' + event\
                           + '_PEDataRelease_mixed_' + model + '.h5'
                file = h5py.File(directory + filename, 'r')
            except OSError:
                print(f'No file found for event \'{event}\' in directory'\
                      f' \'{directory}\'.')

        # Extract relevant data from file
        try:
            Xph = create_datamatrix(
                file['C01:IMRPhenomXPHM/posterior_samples'], 
                params
                )
        except KeyError:
            Xph = create_datamatrix(
                file['C01:IMRPhenomXPHM:HighSpin/posterior_samples'], 
                params
                )
        
        Xse = create_datamatrix(
            file['C01:SEOBNRv4PHM/posterior_samples'], 
            params
            )


        # Apply optional corrections to data
        if centered:
            Xph -= np.mean(Xph, axis = 0)
            Xse -= np.mean(Xse, axis = 0)
        
        if normed:
            Xph /= np.std(Xph, axis = 0)
            Xse /= np.std(Xse, axis = 0)


        # Deal with axes, first special case sample size and then regular case
        if 'sample_size' in axes:
            x += [Xph.shape[0]]
            y += [Xse.shape[0]]
        else:
            x += [np.percentile(
                file['C01:Mixed/posterior_samples'][axes[0]], 
                50
                )]
            y += [np.percentile(
                file['C01:Mixed/posterior_samples'][axes[1]], 
                50
                )]

        
        # Calculate value of chosen criterion for all parameters, current event
        if criterion == 'jsd':
            # Jensen-Shannon divergence
            critvals += [np.amax([jsd(Xph[:, j], Xse[:, j]) 
                                  for j in range(parlen)])]

            # Set title of plot
            # title = 'Maximum JSD'

        elif criterion == 'meandiff':
            # Mean difference in units of average standard deviation            
            critvals += [np.amax(mean_criterion(Xph, Xse))]

            # Set title of plot
            # title = 'Maximum mean difference/average stddev'

        elif criterion == 'mediandiff':
            # Median difference in units of average credible interval
            critvals += [np.amax(median_criterion(Xph, Xse, 34.135))]

            # Set title of plot
            # title ='Maximum median difference/average credible interval'

        else:
            raise ValueError('Invalid criterion.')


    # Make plot
    fig, ax = plt.subplots(figsize = (6, 6))

    critvals = np.array(critvals)

    # Create mask which makes it easy to choose different markers for points
    mask = np.where(critvals <= threshold,
                    'threshold\n passed', 'threshold\n failed')

    # Plot median values, color based on criterion values in original and
    # change style based on mask
    sns.scatterplot(x = x, y = y, hue = critvals, style = mask, 
                    style_order = ['threshold\n failed', 'threshold\n passed'], 
                    s = 80, palette = palette, ax = ax)

    ax.legend(bbox_to_anchor = (1.0, 0.5), loc = 'center left')
    # ax.legend(bbox_to_anchor = (0.5, 1.02), loc = 'lower center', ncol = 3)

    # ax.set_title(title)
    ax.grid(True)

    # First again special case with sample size, then case where
    # displayparams are given; else take just axes names themselves
    if 'sample_size' in axes:
        ax.set_xlabel('Sample size Phenom')
        ax.set_ylabel('Sample size EOB')
    elif displayparams is not None:
        ax.set_xlabel(displayparams[axes[0]])
        ax.set_ylabel(displayparams[axes[1]])
    else:
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])

    if limits is not None:
        # It is possible that only x- or y-limits are given, so
        # accessing the other ones would cause an error without try
        try:
            ax.set_xlim(limits['x'][0], limits['x'][1])
        except KeyError:
            pass

        try:
            ax.set_ylim(limits['y'][0], limits['y'][1])
        except KeyError:
            pass
    
    # Create pdf of plot
    if save:
        # If path already contains a name + data format, this name is
        # taken; otherwise, an automatic one is created
        if '.' in path[-4:]:
            plt.savefig(path, bbox_inches = 'tight')
        else:
            name = f'event_correlation_2D_{model}_crit{criterion}_{str(axes)}'
            # if limits:
            #     name += f'_limits{str(limits)}'
            # gives error due to ':'; maybe do 'limitstr =' after 
            # check for not None, then append in try?

            if normed:
                name += '_normed'
            if centered:
                name += 'centered'
            
            plt.savefig(path + name + '.pdf', bbox_inches = 'tight')
    
    plt.show()


def param_assessment(params: list, events: list, criterion: str = 'jsd',
                     threshold: float = 0.05, normed: bool = False,
                     centered: bool = False, model: str = 'nocosmo',
                     displayparams: dict = None, directory: str = '',
                     exclude: list = None):
    """
    Computes the number of events where the differences in the 1D
    posteriors of IMRPhenomXPHM and SEOBNRv4PHM exceed certain
    thresholds for selected parameters.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found
          e.g. using the find_datasets function from the gwosc.datasets
          package.
        - criterion (str, optional, default = 'jsd'): specifies the
          criterion used to measure the differences. Can have values
          'jsd' (for Jensen-Shannon divergence), 'meandiff' (for mean
          difference in units of the average standard deviation) or
          'mediandiff' (for median difference in units of the average
          credible interval).
        - threshold (float, optional, default = 0.05): value used as
          crossing between criterion passed/ not passed. Default
          corresponds to JSD of 50% mean shifted normal distributions.
        - normed (boolean, optional, default = False): specifies if
          posteriors are normalized to have a standard deviation of 1.
        - centered (boolean, optional, default = False): specifies if
          posteriors are normalized to have a mean of 0.
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - displayparams (dictionary, optional, default = None):
          contains parameters from params as keys and display names as
          values for better looking output. If None, the elements from
          params are taken.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from events.

    Returns:
        - failed (dict): contains parameters as keys and information on
          how many events failed the threshold as value.
    """


    # Check conditions
    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')
    
    if exclude:
        for event in exclude:
            events.remove(event)


    # Store frequently used constants and relevant results            
    n = len(events)
    results = event_check(params, events, criterion = criterion, model = model,
                          plot = False, normed = normed, centered = centered,
                          directory = directory)

    # Create dictionary that will be output
    failed = {param: 0 for param in params}

    # Go through events and count how many fail threshold for each parameter
    for i in range(n):
        for j, param in enumerate(params):
            if results[i][j] > threshold:
                failed[param] += 1
    
    # Convert output into more meaningful form
    if displayparams is None:
        failed = {param: f'{failed[param]}/ {n} ('\
            f'{round(failed[param] / n * 100)}\%)' for param in failed}
    else:
        failed = {displayparams[param]: f'{failed[param]}/ {n} '\
            f'({round(failed[param] / n * 100)}\%)' for param in failed}

    return failed


def generate_statistics(params: list, latexparams: dict, eventlists: list,
                        criteria: dict, thresholds: list,
                        model: str = 'nocosmo', header: str = None,
                        caption: str = None, label: str = None,
                        directory: str = '', exclude: list = None,
                        save: bool = False, path: str = ''):
    """
    Generates LaTeX code for a table which contains a summary of the
    information on differences between IMRPhenomXPHM and SEOBNRv4PHM 1D
    posteriors for multiple GW events according to selected criteria.
    This table is either printed or saved to a text file. These
    statistics can be created for multiple eventlists, e.g. to compare
    the respective summaries. The results will be stacked in rows.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - latexparams (dictionary, optional, default = None): contains
          parameters from params as keys and LaTeX code for their names
          as values. If None, the elements from params are taken.
        - eventlists (list or array-like): list of lists which contain
          names of GW events to analyze (e.g. GW150914_095045). Such a
          list can be found e.g. using the find_datasets function from
          the gwosc.datasets package.
        - criteria (dict): specifies the criteria making up columns of
          the table. Has to have keys 'criterion' (string), 'normed'
          (boolean), 'centered' (boolean) where 'criterion' can have
          values allowed as criterion in event_check. These are 'jsd'
          (for Jensen-Shannon divergence), 'meandiff' (for mean
          difference in units of the average standard deviation) or
          'mediandiff' (for median difference in units of the average
          credible interval).
        - thresholds (list or array-like): contains values used to
          determine if the criterion at the corresponding index is
          passed/ not passed.
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - header (string, optional, default = None): text to put in
          header of the table. If None, an automatic one is created.
        - caption (string, optional, default = None): text to put in
          caption of the table. If None, no caption is inserted.
        - label (string, optional, default = None): text to put in
          label of the table. If None, no label is inserted.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from eventlists.
        - save (boolean, optional, default = False): if True, the table
          gets saved as a text file to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'results/table.txt', it is assumed
          to also contain a file name, so in the example a txt-file
          named 'table' will be saved to the subfolder 'results'.

    Returns:
        - None
    """


    # Check conditions
    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')

    # Convert certain lists to numpy-arrays for convenient commands
    params = np.array(params)
    
    # Store frequently used constants
    n = len(eventlists)
    parlen = len(params)
    critlen = len(criteria)


    # Start string which will form table in the end
    tablestr = '\\begin{table}\n\centering\n\small\n\n\\begin{tabular}{'\
               + (critlen + 2) * 'c ' + '}\n\\toprule\n\\toprule\n'
    # The command \small is necessary because table gets too wide otherwise 
    # (even smaller is \footnotesize, \resizebox also convenient).


    # Adds header to table (automatically of None is given)
    if header:
        tablestr += '\n' + header
    else:
        header = '\n'
        for crit in criteria:
            header += ' & Criterion ' + str(crit)
        header += '\\\\'
        
        tablestr += header


    for i in range(n):
        # Add row separation and entry for first column
        tablestr += '\n\n\midrule\n\multirow{' + str(parlen) + '}*{Eventlist '\
                    + str(i) + '}'

        # Calculate how events from respective eventlist perform under criteria
        data = [param_assessment(params, eventlists[i],
                                 criterion = crit['criterion'],
                                 threshold = thresholds[c],
                                 normed = crit['normed'],
                                 centered = crit['centered'], model = model,
                                 directory = directory, exclude = exclude)
                for c, crit in enumerate(criteria)]
        

        # Add multiple rows with results for eventlist
        for j, param in enumerate(params):
            # Color certain rows for better looking table (instead of adding
            # \midrule at the beginning of rows)
            if j % 2 == 0:#if (parlen * i + j) % 2 == 0:
                tablestr += '\n & ' + latexparams[param]
            else:
                tablestr += '\n\\rowcolor{gray!36} & ' + latexparams[param]

            # Add results to row
            for k in range(critlen):
                tablestr += ' & ' + data[k][param]

            # New row for new parameter
            tablestr += '\\\\'


    # Add (possibly empty) caption and label
    tablestr += '\n\n\\bottomrule\n\\bottomrule\n\\end{tabular}\n\n\caption{'
    
    if caption:
        tablestr += caption
    
    tablestr += '}\n\label{tab:'
    
    if label:
        tablestr += label
    
    tablestr += '}\n\\end{table}'


    # Create txt file containing the table or print it (e.g. for copy + paste)
    if save:
        # If path already contains a name + data format, this name is
        # taken; otherwise, an automatic one is created
        if '.' in path[-4:]:
            with open(path, 'w') as f:
                f.write(tablestr)
        else:
            # Create informative name
            txtname = path + 'event_statistics_'

            for crit in criteria:
                txtname += str(crit['criterion'])
                if crit['normed']:
                    txtname += 'normed'
                if crit['centered']:
                    txtname += 'centered'
            
            # Write table to file
            with open(path + txtname + '.txt', 'w') as f:
                f.write(tablestr)
    else:
        print(tablestr)


def covariance_avg(events: list, params: list, basis: str = None,
                   model: str = 'nocosmo', normed: bool = False,
                   centered = False, abs: bool = False, plot = True,
                   displayparams: dict = None, directory: str = '',
                   save: bool = False, path: str = ''):
    """
    Computes the average covariance matrix of posteriors from GW events
    and the transformation which diagonalizes this matrix. This is
    either returned or visualized.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found
          e.g. using the find_datasetsfunction from the gwosc.datasets
          package.
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - basis (string, optional, default = None): name of basis
          spanned by eigenvectors of covariance matrix of one of the
          posteriors. Can have values 'Phenom' (abbreviation of
          IMRPhenomXPHM) and 'EOB' (abbreviation of SEOBNRv4PHM).
          If None, the results from both models will be used.
        - normed (boolean, optional, default = False): specifies if
          posteriors are normalized to have a standard deviation of 1.
        - centered (boolean, optional, default = False): specifies if
          posteriors are normalized to have a mean of 0.
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - plot (boolean, optional, default = True): specifies if output
          is a plot or the list of results.
        - threshold (float, optional, default = 0.05): value used to
          draw a line in the plot, corresponding to the crossing
          between criterion passed/ not passed. Default corresponds to
          JSD of 50% mean shifted normal distributions.
        - displayparams (dictionary, optional, default = None):
          contains parameters from params as keys and display names for
          lots as values (e.g. LaTeX code). If None, the elements from
          params are taken.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from events.
        - save (boolean, optional, default = False): if True, the plot
          gets saved to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'pictures/photo.png', it is assumed
          to also contain a file name, so in the example a png-file
          named 'photo' will be saved to the subfolder 'pictures'.

    Returns:
        - If basis is None: either a list containing the average
          covariance matrix of the Mixed posteriors of all elements of
          events and the matrix of eigenvectors computed from this
          average covariance matrix or None. If basis is not None: a
          similar list, but now the quantities are computed separately
          for IMRPhenomXPHM and SEOBNRv4PHM posteriors (order of return
          is covariance matrices for Phenom, EOB and then the
          transformations for Phenom, EOB) or None.
    """


    # Check conditions
    if (basis and (basis != 'Prior') and (basis != 'Phenom')
        and (basis != 'EOB')):
        raise ValueError('Basis has to be either \'None\', \'Prior\','\
                         ' \'Phenom\' or \'EOB\'.')
        
    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')

    if displayparams is None:
        displayparams = params
    else:
        displayparams = [displayparams[param] for param in params]
    

    if basis is None:
        parlen = len(params)

        covsumph = np.zeros((parlen, parlen))
        covsumse = np.zeros((parlen, parlen))

        for event in events:
            try:
                filename = 'IGWN-GWTC3p0-v1-' + event\
                           + '_PEDataRelease_mixed_' + model + '.h5'
                file = h5py.File(directory + filename, 'r')
            except OSError:
                try:
                    filename = 'IGWN-GWTC2p1-v2-' + event\
                               + '_PEDataRelease_mixed_' + model + '.h5'
                    file = h5py.File(directory + filename, 'r')
                except OSError:
                    print(f'No file found for event \'{event}\' in directory'\
                          f' \'{directory}\'.')

            try:
                fileXph = create_datamatrix(
                    file['C01:IMRPhenomXPHM/posterior_samples'],
                    params
                    )
            except KeyError:
                fileXph = create_datamatrix(
                    file['C01:IMRPhenomXPHM:HighSpin/posterior_samples'],
                    params
                    )

            fileXse = create_datamatrix(
                file['C01:SEOBNRv4PHM/posterior_samples'],
                params
                )

            if normed:
                covph = np.corrcoef(fileXph, rowvar = False)
                covse = np.corrcoef(fileXse, rowvar = False)
            else:
                covph = np.cov(fileXph, rowvar = False)
                covse = np.cov(fileXse, rowvar = False)

            if abs:
                covsumph += np.abs(covph)
                covsumse += np.abs(covse)
            else:
                covsumph += covph
                covsumse += covse

        covsumph /= len(events)
        covsumse /= len(events)

        # Compute eigensystem average covariance matrix
        evals_ph, Aph = np.linalg.eigh(covsumph)
        evals_se, Ase = np.linalg.eigh(covsumse)

        # Sort eigenvalues and -vectors in descending order
        perm_ph = np.flip(evals_ph.argsort())
        evals_ph = evals_ph[perm_ph]
        Aph = Aph[:, perm_ph]
        perm_se = np.flip(evals_se.argsort())
        evals_se = evals_se[perm_se]
        Ase = Ase[:, perm_se]


        if plot:
            fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize = (20, 14))
            # (24, 14) if 'of variance' is in

            cov_df_ph = pd.DataFrame(
                np.round(covsumph, 3),
                columns = displayparams,
                index = displayparams
                )
            cov_df_se = pd.DataFrame(
                np.round(covsumse, 3),
                columns = displayparams,
                index = displayparams
                )

            sns.heatmap(cov_df_ph.abs(), annot = cov_df_ph, fmt = 'g', 
                        ax = axs[0, 0])
            sns.heatmap(cov_df_se.abs(), annot = cov_df_se, fmt = 'g', 
                        ax = axs[0, 1])

            # axs[0].set_suptitle('Covariance matrices')
            axs[0, 0].set_title('Phenom')
            axs[0, 1].set_title('EOB')


            pcconvert_ph = 1 / np.sum(evals_ph) * 100  # Conversion factor
            percents_ph = [f'{i + 1} ({evals_ph[i] * pcconvert_ph: .2f}%)'
                           for i in range(len(evals_ph))]  # % of variance
            eigenvec_df_ph = pd.DataFrame(
                np.round(Aph.T, 2),
                columns = displayparams,
                index = percents_ph
                )
            
            pcconvert_se = 1 / np.sum(evals_se) * 100  # Conversion factor
            percents_se = [f'{i + 1} ({evals_se[i] * pcconvert_se: .2f}%)'
                           for i in range(len(evals_se))] # % of variance
            eigenvec_df_se = pd.DataFrame(
                np.round(Ase.T, 2),
                columns = displayparams,
                index = percents_se
                )
        
            sns.heatmap(eigenvec_df_ph.abs(), vmin = 0, vmax = 1,
                        annot = eigenvec_df_ph, fmt = 'g', ax = axs[1, 0])
            sns.heatmap(eigenvec_df_se.abs(), vmin = 0, vmax = 1,
                        annot = eigenvec_df_se, fmt = 'g', ax = axs[1, 1])
            # Coloring is based on absolute value, but values with signs are
            # written into the cells (strength more important than sign).
            # Color palette is rocket from seaborn (default)


            # Create pdf of plot
            if save:
                # If path already contains a name + data format, this name is
                # taken; otherwise, an automatic one is created
                if '.' in path[-4:]:
                    plt.savefig(path, bbox_inches = 'tight')
                else:
                    name = f'covariance_avg_{model}_{str(params)}'

                    if normed:
                        name += '_normed'
                    if centered:
                        name += 'centered'
                    
                    plt.savefig(path + name + '.pdf', bbox_inches = 'tight')
                    
            plt.show()
        else:
            return [covsumph, covsumse, Aph, Ase]
    else:
        parlen = len(params)

        covsum = np.zeros((parlen, parlen))

        for event in events:
            try:
                filename = 'IGWN-GWTC3p0-v1-' + event\
                           + '_PEDataRelease_mixed_' + model + '.h5'
                file = h5py.File(directory + filename, 'r')
            except OSError:
                try:
                    filename = 'IGWN-GWTC2p1-v2-' + event\
                               + '_PEDataRelease_mixed_' + model + '.h5'
                    file = h5py.File(directory + filename, 'r')
                except OSError:
                    print(f'No file found for event \'{event}\' in directory'\
                          f' \'{directory}\'.')


            if basis == 'Prior':
                try:
                    fileX = create_datamatrix(
                        file['C01:IMRPhenomXPHM/priors/samples'],
                        params
                        )
                except KeyError:
                    fileX = create_datamatrix(
                        file['C01:IMRPhenomXPHM:HighSpin/priors/samples'],
                        params
                        )
            elif basis == 'Phenom':
                try:
                    fileX = create_datamatrix(
                        file['C01:IMRPhenomXPHM/posterior_samples'],
                        params
                        )
                except KeyError:
                    fileX = create_datamatrix(
                        file['C01:IMRPhenomXPHM:HighSpin/posterior_samples'],
                        params
                        )
            else:
                fileX = create_datamatrix(
                    file['C01:SEOBNRv4PHM/posterior_samples'],
                    params
                    )

            if normed:
                cov = np.corrcoef(fileX, rowvar = False)
            else:
                cov = np.cov(fileX, rowvar = False)

            if abs:
                covsum += np.abs(cov)
            else:
                covsum += cov

        covsum /= len(events)

        # Compute eigensystem average covariance matrix
        evals, A = np.linalg.eigh(covsum)

        # Sort eigenvalues and -vectors in descending order
        perm = np.flip(evals.argsort())
        evals = evals[perm]
        A = A[:, perm]


        if plot:
            fig, axs = plt.subplots(ncols = 2, figsize = (20, 7))
            # (24, 7) in case 'of variance' is in

            cov_df = pd.DataFrame(
                np.round(covsum, 3),
                columns = displayparams,
                index = displayparams
                )

            sns.heatmap(cov_df.abs(), annot = cov_df, fmt = 'g', ax = axs[0])

            # axs[0].set_suptitle('Covariance matrices')
            axs[0].set_title(basis)


            pcconvert = 1 / np.sum(evals) * 100 # Conversion factor
            percents = [f'{i + 1} ({evals[i] * pcconvert: .2f}%)'
                        for i in range(len(evals))] # % of variance
            eigenvec_df = pd.DataFrame(
                np.round(A.T, 2),
                columns = displayparams,
                index = percents
                )
        
            sns.heatmap(eigenvec_df.abs(), vmin = 0, vmax = 1,
                        annot = eigenvec_df, fmt = 'g', ax = axs[1])
            # Coloring is based on absolute value, but values with signs are
            # written into the cells (strength more important than sign).
            # Color palette is rocket from seaborn (default)
            

            # Create pdf of plot
            if save:
                # if path already contains a name + data format, this name is
                # taken; otherwise, an automatic one is created
                if '.' in path[-4:]:
                    plt.savefig(path, bbox_inches = 'tight')
                else:
                    name = f'covariance_avg_{basis}_{model}_{str(params)}'

                    if normed:
                        name += '_normed'
                    if centered:
                        name += 'centered'
                    
                    plt.savefig(path + name + '.pdf', bbox_inches = 'tight')

            plt.show()
        else:
            return [covsum, A]



#%% ---------- Functions for analysis of PCs ----------

def event_check_PCs(basis: str, params: list, events: list,
                    criterion: str = 'jsd', normed: bool = False,
                    centered: bool = False, model: str = 'nocosmo',
                    plot: bool = True, threshold: float = 0.05,
                    directory: str = '', exclude: list = None, limits = None,
                    save: bool = False, path: str = ''):
    """
    Computes the difference between the Principal Components (PCs) of
    IMRPhenomXPHM and SEOBNRv4PHM 1D posteriors for multiple GW events
    according to a selected criterion. This is either returned or
    visualized.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - basis (string): name of basis spanned by eigenvectors of
          covariance matrix of one of the posteriors. Can have values
          'Phenom' (abbreviation of IMRPhenomXPHM) and 'EOB'
          (abbreviation of SEOBNRv4PHM).
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found
          e.g. using the find_datasets function from the gwosc.datasets
          package.
        - criterion (str, optional, default = 'jsd'): specifies the
          criterion used to measure the differences. Can have values
          'jsd' (for Jensen-Shannon divergence), 'jsd2' (Jensen-Shannon
          divergence of normalized, mean-centered distributions),
          'meandiff' (for mean difference in units of the average
          standard deviation) or 'mediandiff' (for median difference in
          units of the average credible interval).
        - normed (boolean, optional, default = False): specifies if
          posteriors are normalized to have a standard deviation of 1
          (before the data is transformed; standard deviation is meant
          with respect to the chosen parameters).
        - centered (boolean, optional, default = False): specifies if
          posteriors are normalized to have a mean of 0 (the Principal
          Components will then also have a mean of 0).
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - plot (boolean, optional, default = True): specifies if output
          is a plot or the list of results.
        - threshold (float, optional, default = 0.05): value used to
          draw a line in the plot, corresponding to the crossing
          between criterion passed/ not passed. Default corresponds to
          JSD of 50% mean shifted normal distributions.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from events.
        - limits (tuple or list with two elements, optional,
          default = None): sets specific limits on y-axis.
        - save (boolean, optional, default = False): if True, the plot
          gets saved to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'pictures/photo.png', it is assumed
          to also contain a file name, so in the example a png-file
          named 'photo' will be saved to the subfolder 'pictures'.

    Returns:
        - Either a list containing a list with differences in the
          parameter posteriors for every GW event in events or None.
    """


    # Check conditions
    if (basis != 'Phenom') and (basis != 'EOB'):
        raise ValueError('Basis has to be either \'Phenom\' or \'EOB\'.')

    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')
    
    if exclude:
        for event in exclude:
            events.remove(event)
            
    
    # Store frequently used constants
    n = len(events)
    parlen = len(params)
    
    # Create list to append results from events
    critvals = []
    
    for event in events:
        # Open file containing event data
        try:
            filename = 'IGWN-GWTC3p0-v1-' + event + '_PEDataRelease_mixed_'\
                       + model + '.h5'
            file = h5py.File(directory + filename, 'r')
        except OSError:
            try:
                filename = 'IGWN-GWTC2p1-v2-' + event\
                           + '_PEDataRelease_mixed_' + model + '.h5'
                file = h5py.File(directory + filename, 'r')
            except OSError:
                print(f'No file found for event \'{event}\' in directory'\
                      f' \'{directory}\'.')

        # Extract relevant data from file
        try:
            Xph = create_datamatrix(
                file['C01:IMRPhenomXPHM/posterior_samples'],
                params
                )
        except KeyError:
            Xph = create_datamatrix(
                file['C01:IMRPhenomXPHM:HighSpin/posterior_samples'],
                params
                )
            
        Xse = create_datamatrix(
            file['C01:SEOBNRv4PHM/posterior_samples'],
            params
            )


        # Transform data into specified basis
        if basis == 'Phenom':
            eventpca = pca(Xse, basis = Xph, normed = normed,
                           centered = centered)
            Zph = eventpca.get_basisPCs()
            Zse = eventpca.get_PCs()
        else:
            eventpca = pca(Xph, basis = Xse, normed = normed,
                           centered = centered)
            Zph = eventpca.get_PCs()
            Zse = eventpca.get_basisPCs()
        
        
        # Calculate value of chosen criterion for all parameters and chosen event
        if criterion == 'jsd':
            # Jensen-Shannon divergence
            critvals += [[jsd(Zph[:, j], Zse[:, j]) for j in range(parlen)]]

            # Title of plot which is generated in some cases
            # title = 'JSD for original parameters'

        elif criterion == 'jsd2':
            # Jensen-Shannon divergence of normalized, mean-centered input
            Zph -= np.mean(Zph, axis = 0)
            Zse -= np.mean(Zse, axis = 0)

            Zph /= np.std(Zph, axis = 0)
            Zse /= np.std(Zse, axis = 0)

            critvals += [[jsd(Zph[:, j], Zse[:, j]) for j in range(parlen)]]

            # Title of plot which is generated in some cases
            # title = 'JSD for normalized, centered parameters'

        elif criterion == 'meandiff':
            # Mean difference in units of average standard deviation            
            critvals += [mean_criterion(Zph, Zse)]

            # Title of plot which is generated in some cases
            # title = 'Mean difference/average stddev for original parameters'

        elif criterion == 'mediandiff':
            # Median difference in units of average credible interval
            critvals += [median_criterion(Zph, Zse, 34.135)]
            
            # Title of plot which is generated in some cases
            # title = 'Median difference/average credible interval for'\
            #         ' original parameters'
            
        else:
            raise ValueError('Invalid criterion')


    # Return list with criterion values or plot of them
    if plot:

        fig, ax = plt.subplots(figsize = (n / 2, 6))

        # Create numpy-array for convenient access to columns
        critvals = np.array(critvals)

        x = [i for i in range(n)]

        for k in range(parlen):
            ax.plot(x, critvals[:, k], 'o', label = f'PC {str(k + 1)}')
        
        
        ax.axline((0, threshold), (n - 1, threshold), color = 'r')

        ax.legend(bbox_to_anchor = (1.0, 0.5), loc = 'center left')
        # ax.set_title(title)
        ax.grid(axis = 'y')
        
        # set number of ticks for x-axis and label ticks
        ax.set_xticks(x, events, rotation = 45, horizontalalignment = 'right',
            rotation_mode = 'anchor')#rotation = 'vertical')

        if limits:
            ax.set_ylim(limits[0], limits[1])
        
        # Create pdf of plot
        if save:
            # If path already contains a name + data format, this name is
            # taken; otherwise, an automatic one is created
            if '.' in path[-4:]:
                plt.savefig(path, bbox_inches = 'tight')
            else:
                name = f'eventplot_PCs_{basis}_{model}_crit{criterion}'\
                       f'_{str(params)}'
                # if limits:
                #     name += f'_limits{limits}'
                
                if normed:
                    name += '_normed'
                if centered:
                    name += 'centered'
                
                plt.savefig(path + name + '.pdf', bbox_inches = 'tight')
    
    else:
        return critvals


def generate_table_PCs(basis: str, params: list, events: list, criteria: dict,
                       thresholds: list,
                       exceedvals: dict = {'red': 2, 'yellow': 1},
                       badevents: list = None, goodevents: list = None,
                       model: str = 'nocosmo', header: str = None,
                       caption: str = None, label: str = None,
                       directory: str = '', exclude: list = None,
                       save: bool = False, path: str = ''):
    """
    Generates Latex code for a table which contains information on
    differences between the Principal Components (PCs) of IMRPhenomXPHM
    and SEOBNRv4PHM 1D posteriors for multiple GW events according to
    selected criteria. This table is either printed or saved to a text
    file.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - basis (string): name of basis spanned by eigenvectors of
          covariance matrix of one of the posteriors. Can have values
          'Phenom' (abbreviation of IMRPhenomXPHM) and 'EOB'
          (abbreviation of SEOBNRv4PHM).
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found
          e.g. using the find_datasets function from the gwosc.datasets
          package.
        - criteria (dict): specifies the criteria making up columns of
          the table. Has to have keys 'criterion' (string), 'normed'
          (boolean), 'centered' (boolean) where 'criterion' can have
          values allowed as criterion in event_check. These are 'jsd'
          (for Jensen-Shannon divergence), 'jsd2' (Jensen-Shannon
          divergence of normalized, mean-centered distributions),
          'meandiff' (for mean difference in units of the average
          standard deviation) or 'mediandiff' (for median difference
          in units of the average credible interval).
        - thresholds (list or array-like): contains values used to
          determine if the criterion at the corresponding index is
          passed/ not passed.
        - exceedvals (dict, optional, default = {'red': 2,
          'yellow': 1}): specifies how much parameters are allowed to
          exceed criteria before modest (yellow) or high (red)
          significance is marked. Must contain keys 'red' and 'yellow'
          with the corresponding keys being integers (how many
          parameter posteriors are allowed to violate).
        - badevents (list or array-like, optional, default = None):
          enables highlighting of events with known bad agreement in
          red (or any other color encoding). If None, no events are
          highlighted.
        - goodevents (list or array-like, optional, default = None):
          enables highlighting of events with known good agreement in
          green (or any other color encoding). If None, no events are
          highlighted.
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - header (string, optional, default = None): text to put in
          header of the table. If None, an automatic one is created.
        - caption (string, optional, default = None): text to put in
          caption of the table. If None, no caption.
        - label (string, optional, default = None): text to put in
          label of the table. If None, no label.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from events.
        - save (boolean, optional, default = False): if True, the table
          gets saved as a text file to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'results/table.txt', it is assumed
          to also contain a file name, so in the example a txt-file
          named 'table' will be saved to the subfolder 'results'.

    Returns:
        - None
    """
    

    # check conditions
    if (basis != 'Phenom') and (basis != 'EOB'):
        raise ValueError('Basis has to be either \'Phenom\' or \'EOB\'.')

    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')

    if exclude:
        for event in exclude:
            events.remove(event)


    # calculate how events perform under chosen criteria
    data = [event_check_PCs(basis, params, events,
                            criterion = crit['criterion'],
                            normed = crit['normed'],
                            centered = crit['centered'], model = model,
                            plot = False, directory = directory,
                            exclude = exclude) for crit in criteria]
    
    # Convert certain lists to numpy-arrays for convenient commands
    data = np.array(data)
    params = np.array(params)
    
    # Store frequently used constants
    critlen = len(criteria)


    # Start string which will form table in the end
    tablestr = '\\begin{table}\n\centering\n\small\n\n\\begin{tabular}{'\
               + (critlen + 1) * 'c ' + '}'
    # The command \small is necessary because table gets too wide otherwise 
    # (even smaller is \footnotesize, \resizebox also convenient).


    # Adds header to table (automatically of None is given)
    if header:
        tablestr += '\n' + header
    else:
        header = '\n'
        for crit in criteria:
            header += ' & Criterion ' + str(crit)
        header += '\\\\'
        
        tablestr += header


    for i, event in enumerate(events):
        # Add row separation and entry for first column
        if (badevents is not None) and (event in badevents):
            tablestr += '\n\midrule\n\cellcolor{red!36} ' + event[:8] + '\\_'\
                        + event[9:] + ' ' # only _ produces LaTeX error
        elif (goodevents is not None) and (event in goodevents):
            tablestr += '\n\midrule\n\cellcolor{green!36} ' + event[:8]\
                        + '\\_' + event[9:] + ' '
        else:
            tablestr += '\n\midrule\n' + event[:8] + '\\_' + event[9:] + ' '
        # Instead of '\midrule', '\hline' could be chosen and the same is true
        # for r'\_' and r'\textunderscore', r'\verb|_|'

        
        # Alternative: different cellcolors for events instead of rules
        # if i % 2 == 0:
        #     tablestr += '\n\n' + event[:8] + '\_' + event[9:]
        # else:
        #     tablestr += '\n\n\cellcolor{gray!36} ' + event[:8] + '\_'\
        #                 + event[9:]


        # Add results for event to this row
        for j in range(critlen):
            tablestr += '& '

            # Determine number of parameters for which threshold is exceeded
            numb = len(data[j][i][data[j][i] >= thresholds[j]])
            
            # Based on numb, set cellcolor and add parameters which failed
            if numb >= exceedvals['red']:
                tablestr += '\cellcolor{red!36} '

                for k, boolval in enumerate(data[j][i] >= thresholds[j]):
                    if boolval:
                        tablestr += str(k + 1) + ' '
            elif numb >= exceedvals['yellow']:
                tablestr += '\cellcolor{yellow!36} '

                for k, boolval in enumerate(data[j][i] >= thresholds[j]):
                    if boolval:
                        tablestr += str(k + 1) + ' '
            else:
                tablestr += '\cellcolor{green!36} '

        # New row for new event
        tablestr += '\\\\'

    tablestr += '\n\\end{tabular}\n\n\caption{'
    

    # Add (possibly empty) caption and label
    if caption:
        tablestr += caption
    
    tablestr += '}\n\label{tab:'
    
    if label:
        tablestr += label
    
    tablestr += '}\n\\end{table}'


    # Create txt file containing the table or print it (e.g. for copy + paste)
    if save:
        # If path already contains a name + data format, this name is
        # taken; otherwise, an automatic one is created
        if '.' in path[-4:]:
            with open(path, 'w') as f:
                f.write(tablestr)
        else:
            # Create informative name
            txtname = path + 'eventdata_PCs_' + basis

            for crit in criteria:
                txtname += str(crit['criterion'])
                if crit['normed']:
                    txtname += 'normed'
                if crit['centered']:
                    txtname += 'centered'
            
            # Write table to file
            with open(path + txtname + '.txt', 'w') as f:
                f.write(tablestr)
    else:
        print(tablestr)


def param_assessment_PCs(basis: str, params: list, events: list,
                         criterion: str = 'jsd', threshold: float = 0.05,
                         normed: bool = False, centered: bool = False,
                         model: str = 'nocosmo', directory: str = '',
                         exclude: list = None):
    """
    Computes for how many events the Principal Components (PCs) of the
    1D posteriors of IMRPhenomXPHM and SEOBNRv4PHM exceed certain
    differences for certain parameters.

    Parameters:
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found
          e.g. using the find_datasets function from the gwosc.datasets
          package.
        - criterion (str, optional, default = 'jsd'): specifies the
          criterion used to measure the differences. Can have values
          'jsd' (for Jensen-Shannon divergence), 'meandiff' (for mean
          difference in units of the average standard deviation) or
          'mediandiff' (for median difference in units of the average
          credible interval).
        - threshold (float, optional, default = 0.05): value used as
          crossing between criterion passed/ not passed. Default
          corresponds to JSD of 50% mean shifted normal distributions.
        - normed (boolean, optional, default = False): specifies if
          posteriors are normalized to have a standard deviation of 1.
        - centered (boolean, optional, default = False): specifies if
          posteriors are normalized to have a mean of 0.
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from events.

    Returns:
        - failed (dict): contains PC numbers as keys and information on
          how many events failed the threshold as value.
    """
    

    # Check conditions
    if (basis != 'Phenom') and (basis != 'EOB'):
        raise ValueError('Basis has to be either \'Phenom\' or \'EOB\'.')

    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')
    
    if exclude:
        for event in exclude:
            events.remove(event)


    # Store frequently used constants and relevant results            
    n = len(events)
    parlen = len(params)
    results = event_check_PCs(basis, params, events, criterion = criterion,
                              normed = normed, centered = centered,
                              model = model, plot = False,
                              directory = directory, exclude = exclude)

    # Create dictionary which will be output
    failed = {str(k): 0 for k in range(parlen)}

    # Go through events and count how many fail threshold for each PC
    for i in range(n):
        for j in range(parlen):
            if results[i][j] > threshold:
                failed[str(j)] += 1
    
    # convert output into more meaningful form
    failed = {k: f'{failed[k]}/ {n} ({round(failed[k] / n * 100)}\%)'
              for k in failed}

    return failed


def generate_statistics_PCs(basis: str, params: list, eventlists: list,
                            criteria: dict, thresholds: list,
                            model: str = 'nocosmo', header: str = None,
                            caption: str = None, label: str = None,
                            directory: str = '', exclude: list = None,
                            save: bool = False, path: str = ''):
    """
    Generates LaTeX code for a table which contains a summary of the
    information on differences between the Principal Components of
    IMRPhenomXPHM and SEOBNRv4PHM 1D posteriors for multiple GW events
    according to selected criteria. This table is either printed or
    saved to a text file. These statistics can be created for multiple
    eventlists, e.g. to compare the respective summaries. The results
    will be stacked in rows.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - latexparams (dictionary, optional, default = None): contains
          parameters from params as keys and display names for plots as
          values. If None, the elements from params are taken.
        - eventlists (list or array-like): list of lists which contain
          names of GW events to analyze (e.g. GW150914_095045). Such a
          list can be found e.g. using the find_datasets function from
          the gwosc.datasets package.
        - criteria (dict): specifies the criteria making up columns of
          the table. Has to have keys 'criterion' (string), 'normed'
          (boolean), 'centered' (boolean) where 'criterion' can have
          values allowed as criterion in event_check. These are 'jsd'
          (for Jensen-Shannon divergence), 'jsd2' (Jensen-Shannon
          divergence of normalized, mean-centered distributions),
          'meandiff' (for mean difference in units of the average
          standard deviation) or 'mediandiff' (for median difference
          in units of the average credible interval).
        - thresholds (list or array-like): contains values used to
          determine if the criterion at the corresponding index is
          passed/ not passed.
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - header (string, optional, default = None): text to put in
          header of the table. If None, an automatic one is created.
        - caption (string, optional, default = None): text to put in
          caption of the table. If None, no caption is inserted.
        - label (string, optional, default = None): text to put in
          label of the table. If None, no label is inserted.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from eventlists.
        - save (boolean, optional, default = False): if True, the table
          gets saved as a text file to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'results/table.txt', it is assumed
          to also contain a file name, so in the example a txt-file
          named 'table' will be saved to the subfolder 'results'.

    Returns:
        - None
    """


    # Check conditions
    if (basis != 'Phenom') and (basis != 'EOB'):
        raise ValueError('Basis has to be either \'Phenom\' or \'EOB\'.')

    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')


    # Convert certain lists to numpy-arrays for convenient commands
    params = np.array(params)
    
    # Store frequently used constants
    n = len(eventlists)
    parlen = len(params)
    critlen = len(criteria)


    # Start string which will form table in the end
    tablestr = '\\begin{table}\n\centering\n\small\n\n\\begin{tabular}{'\
               + (critlen + 2) * 'c ' + '}'
    # The command \small is necessary because table gets too wide otherwise 
    # (even smaller is \footnotesize, \resizebox also convenient).


    # Adds header to table (automatically of None is given)
    if header:
        tablestr += '\n' + header
    else:
        header = '\n'
        for crit in criteria:
            header += ' & Criterion ' + str(crit)
        header += '\\\\'
        
        tablestr += header


    for i in range(n):
        # Add row separation and entry for first column
        tablestr += '\n\n\midrule\n\multirow{' + str(parlen) + '}*{Eventlist '\
                    + str(i) + '}'
        
        # Calculate how events from respective eventlist perform under criteria
        data = [param_assessment_PCs(basis, params, eventlists[i],
                                     criterion = crit['criterion'],
                                     threshold = thresholds[c],
                                     normed = crit['normed'],
                                     centered = crit['centered'],
                                     model = model,
                                     directory = directory, exclude = exclude)
                for c, crit in enumerate(criteria)]

        # Add multiple rows with results for eventlist
        for j in range(parlen):
            # Color certain rows for better looking table
            # (instead of adding \midrule at beginning of rows)
            if j % 2 == 0:#if (parlen * i + j) % 2 == 0:
                tablestr += f'\n & PC {j + 1}'
            else:
                tablestr += '\n\\rowcolor{gray!36} & PC ' + str(j + 1)

            # Add results to row
            for k in range(critlen):
                tablestr += ' & ' + data[k][str(j)]

            # New row for new PC
            tablestr += '\\\\'


    # Add (possibly empty) caption and label
    tablestr += '\n\\end{tabular}\n\n\caption{'
    
    if caption:
        tablestr += caption
    
    tablestr += '}\n\label{tab:'
    
    if label:
        tablestr += label
    
    tablestr += '}\n\\end{table}'


    # Create txt file containing the table or print it (e.g. for copy + paste)
    if save:
        # If path already contains a name + data format, this name is
        # taken; otherwise, an automatic one is created
        if '.' in path[-4:]:
            with open(path, 'w') as f:
                f.write(tablestr)
        else:
            # Create informative name
            txtname = path + 'event_statistics_PCs_' + basis

            for crit in criteria:
                txtname += str(crit['criterion'])
                if crit['normed']:
                    txtname += 'normed'
                if crit['centered']:
                    txtname += 'centered'
            
            # Write table to file
            with open(path + txtname + '.txt', 'w') as f:
                f.write(tablestr)
    else:
        print(tablestr)


def compare_quality(params: list, events: list, basis: str = None,
                    criterion: str = 'jsd', normed: bool = False,
                    centered: bool = False, model: str = 'nocosmo',
                    plot: bool = True, threshold: float = 0.05,
                    directory: str = '', exclude: list = None, limits = None,
                    scale: str = None, save: bool = False, path: str = ''):
    """
    Compares the average difference between IMRPhenomXPHM and
    SEOBNRv4PHM 1D posteriors of parameters and PCs for multiple GW
    events computed according to a selected criterion. This is either
    returned or visualized.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found
          e.g. using the find_datasets function from the gwosc.datasets
          package.
        - basis (string, optional, default = None): name of basis
          spanned by eigenvectors of covariance matrix of one of the
          posteriors. Can have values 'Phenom' (abbreviation of
          IMRPhenomXPHM) and 'EOB' (abbreviation of SEOBNRv4PHM).
          If None, the results from both bases will be used.
        - criterion (str, optional, default = 'jsd'): specifies the
          criterion used to measure the differences. Can have values
          'jsd' (for Jensen-Shannon divergence), 'jsd2' (Jensen-Shannon
          divergence of normalized, mean-centered distributions),
          'meandiff' (for mean difference in units of the average
          standard deviation) or 'mediandiff' (for median difference in
          units of the average credible interval).
        - normed (boolean, optional, default = False): specifies if
          posteriors are normalized to have a standard deviation of 1
          (before the data is transformed; standard deviation is meant
          with respect to the chosen parameters).
        - centered (boolean, optional, default = False): specifies if
          posteriors are normalized to have a mean of 0 (the Principal
          Components will then also have a mean of 0).
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - plot (boolean, optional, default = True): specifies if output
          is a plot or the list of results.
        - threshold (float, optional, default = 0.05): value used to
          draw a line in the plot, corresponding to the crossing
          between criterion passed/ not passed. Default corresponds to
          JSD of 50% mean shifted normal distributions.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from events.
        - limits (tuple or list with two elements, optional,
          default = None): sets specific limits on y-axis.
        - scale (string, optional, default = None): if not None, it is
          given as argument to set_yscale().
        - save (boolean, optional, default = False): if True, the plot
          gets saved to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'pictures/photo.png', it is assumed
          to also contain a file name, so in the example a png-file
          named 'photo' will be saved to the subfolder 'pictures'.

    Returns:
        - Either a list containing a list with [mean of difference
          params, standard deviation of difference params, mean of
          difference PCs, standard deviation of difference PCs] for
          every GW event in events or None.
    """


    # Check conditions
    if basis and (basis != 'Phenom') and (basis != 'EOB'):
        raise ValueError('Basis has to be either \'Phenom\' or \'EOB\'.')

    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')

    if exclude:
        for event in exclude:
            events.remove(event)
            
            
    # Store frequently used constants
    n = len(events)
    
    # Generate differences in parameters and PCs for each event#
    if criterion == 'jsd2':
        # Special case because event_check has no option crit = 'jsd2'
        results = event_check(params, events, 'jsd', normed = True,
                              centered = True, model = model, plot = False,
                              threshold = threshold, directory = directory,
                              exclude = exclude, limits = limits)
    else:
        results = event_check(params, events, criterion, normed = False,
                              centered = centered, model = model, plot = False,
                              threshold = threshold, directory = directory,
                              exclude = exclude, limits = limits)
    

    if basis:
        results_PCs = event_check_PCs(basis, params, events, criterion,
                                      normed = normed, centered = centered,
                                      model = model, plot = False,
                                      threshold = threshold,
                                      directory = directory, exclude = exclude,
                                      limits = limits)


        # Compute average value of criterion and deviation
        means = np.mean(results, axis = 1)
        # devs = np.std(results, axis = 1)
        devs = np.transpose([[means[i] - min(results[i]),
                              max(results[i]) - means[i]] for i in range(n)])
        
        means_PCs = np.mean(results_PCs, axis = 1)
        # devs_PCs = np.std(results_PCs, axis = 1)
        devs_PCs = np.transpose([[means_PCs[i] - min(results_PCs[i]),
                                  max(results_PCs[i]) - means_PCs[i]]
                                  for i in range(n)])


        # Return list with average criterion values or plot of them
        if plot:

            fig, ax = plt.subplots(figsize = (n / 2, 6))

            x = [i for i in range(n)]

            ax.errorbar(x, means, yerr = devs, fmt = 'o', capsize = 4,
                        label = 'Original data')

            ax.errorbar(x, means_PCs, yerr = devs_PCs, fmt = 'o', capsize = 4,
                        label = 'PCs')
            
            
            ax.axline((0, threshold), (n, threshold), color = 'r')

            if scale is not None:
                ax.set_yscale(scale)

            ax.legend(bbox_to_anchor = (1.0, 0.5), loc = 'center left')
            # ax.set_title(title)
            ax.grid(axis = 'y')
            
            # Set number of ticks for x-axis and label ticks
            ax.set_xticks(x, events, rotation = 45,
                          horizontalalignment = 'right',
                          rotation_mode = 'anchor')#rotation = 'vertical')

            if limits:
                ax.set_ylim(limits[0], limits[1])
            
            # Create pdf of plot
            if save:
                # If path already contains a name + data format, this name is
                # taken; otherwise, an automatic one is created
                if '.' in path[-4:]:
                    plt.savefig(path, bbox_inches = 'tight')
                else:
                    name = f'compare_quality_{basis}_{model}_crit{criterion}'\
                           f'_{str(params)}'
                    # if limits:
                    #     name += f'_limits{limits}'

                    if normed:
                        name += '_normed'
                    if centered:
                        name += '_centered'
                    if scale is not None:
                      name += '_' + scale
                    
                    plt.savefig(path + name + '.pdf', bbox_inches = 'tight')
        
        else:
            return [means, devs, means_PCs, devs_PCs]
    else:
        results_PCs_ph = event_check_PCs('Phenom', params, events, criterion,
                                         normed = normed, centered = centered,
                                         model = model, plot = False,
                                         threshold = threshold,
                                         directory = directory,
                                         exclude = exclude, limits = limits)
        results_PCs_eob = event_check_PCs('EOB', params, events, criterion,
                                          normed = normed, centered = centered,
                                          model = model, plot = False,
                                          threshold = threshold,
                                          directory = directory,
                                          exclude = exclude, limits = limits)


        # Compute average value of criterion and deviation
        means = np.mean(results, axis = 1)
        # devs = np.std(results, axis = 1)
        devs = np.transpose([[means[i] - min(results[i]),
                              max(results[i]) - means[i]] for i in range(n)])
        
        means_PCs_ph = np.mean(results_PCs_ph, axis = 1)
        devs_PCs_ph = np.transpose([[means_PCs_ph[i] - min(results_PCs_ph[i]),
                                     max(results_PCs_ph[i]) - means_PCs_ph[i]]
                                     for i in range(n)])

        means_PCs_eob = np.mean(results_PCs_eob, axis = 1)
        devs_PCs_eob = np.transpose([[means_PCs_eob[i]
                                      - min(results_PCs_eob[i]),
                                      max(results_PCs_eob[i])
                                      - means_PCs_eob[i]] for i in range(n)])


        # Return list with average criterion values or plot of them
        if plot:

            fig, ax = plt.subplots(figsize = (n / 2, 6))

            x = [i for i in range(n)]

            ax.errorbar(x, means, yerr = devs, fmt = 'o', capsize = 6,
                        label = 'Original data')
            ax.errorbar(x, means_PCs_ph, yerr = devs_PCs_ph, fmt = 'o',
                        capsize = 6, label = 'PCs Phenom')
            ax.errorbar(x, means_PCs_eob, yerr = devs_PCs_eob, fmt = 'o',
                        capsize = 6, label = 'PCs EOB')
            
            
            ax.axline((0, threshold), (n - 1, threshold), color = 'r')

            if scale is not None:
                ax.set_yscale(scale)

            ax.legend(bbox_to_anchor = (1.0, 0.5), loc = 'center left')
            # ax.set_title(title)
            ax.grid(axis = 'y')
            
            # Set number of ticks for x-axis and label ticks
            ax.set_xticks(x, events, rotation = 45,
                          horizontalalignment = 'right',
                          rotation_mode = 'anchor')#rotation = 'vertical')

            if limits:
                ax.set_ylim(limits[0], limits[1])
            
            # Create pdf of plot
            if save:
                # If path already contains a name + data format, this name is
                # taken; otherwise, an automatic one is created
                if '.' in path[-4:]:
                    plt.savefig(path, bbox_inches = 'tight')
                else:
                    name = f'compare_quality_{model}_crit{criterion}'\
                           f'_{str(params)}'
                    # if limits:
                    #     name += f'_limits{limits}'

                    if normed:
                        name += '_normed'
                    if centered:
                        name += '_centered'
                    if scale is not None:
                      name += '_' + scale
                    
                    plt.savefig(path + name + '.pdf', bbox_inches = 'tight')
        
        else:
            return [means, devs, means_PCs_ph, devs_PCs_ph, means_PCs_eob,
                    devs_PCs_eob]


def compare_quality_points(params: list, events: list, basis: str = None,
                           criterion: str = 'jsd', normed: bool = False,
                           centered: bool = False, model: str = 'nocosmo',
                           plot: bool = True, threshold: float = 0.05,
                           directory: str = '', exclude: list = None,
                           limits = None, scale: str = None,
                           save: bool = False, path: str = ''):
    """
    Compares the average difference between IMRPhenomXPHM and
    SEOBNRv4PHM 1D posteriors of parameters and PCs for multiple GW
    events computed according to a selected criterion. This is either
    returned or visualized.

    Remark: this function assumes that each file containing posteriors
    has the name it has in the corresponding data release by the LVK on
    Zenodo. However, this can be modified very easily to an arbitrary
    pattern containing the event names.

    Parameters:
        - params (list or array-like): contains names of parameters to
          take 1D posteriors from. Possible names can be found by
          looking at '.dtype.names' of the posterior_samples attribute
          for the Mixed posterior.
        - events (list or array-like): contains names of GW events to
          analyze (e.g. GW150914_095045). Such a list can be found
          e.g. using the find_datasets function from the gwosc.datasets
          package.
        - basis (string, optional, default = None): name of basis
          spanned by eigenvectors of covariance matrix of one of the
          posteriors. Can have values 'Phenom' (abbreviation of
          IMRPhenomXPHM) and 'EOB' (abbreviation of SEOBNRv4PHM).
          If None, the results from both bases will be used.
        - criterion (str, optional, default = 'jsd'): specifies the
          criterion used to measure the differences. Can have values
          'jsd' (for Jensen-Shannon divergence), 'jsd2' (Jensen-Shannon
          divergence of normalized, mean-centered distributions),
          'meandiff' (for mean difference in units of the average
          standard deviation) or 'mediandiff' (for median difference
          in units of the average credible interval).
        - normed (boolean, optional, default = False): specifies if
          posteriors are normalized to have a standard deviation of 1
          (before the data is transformed; standard deviation is meant
          with respect to the chosen parameters).
        - centered (boolean, optional, default = False): specifies if
          posteriors are normalized to have a mean of 0 (the Principal
          Components will then also have a mean of 0).
        - model (string, optional, default = 'nocosmo'): can have
          values 'nocosmo' (no cosmological model applied) and 'cosmo'
          (with cosmological model applied).
        - plot (boolean, optional, default = True): specifies if output
          is a plot or the list of results.
        - threshold (float, optional, default = 0.05): value used to
          draw a line in the plot, corresponding to the crossing
          between criterion passed/ not passed. Default corresponds to
          JSD of 50% mean shifted normal distributions.
        - directory (string, optional, default = ''): path from
          directory where program is stored to where posteriors are
          stored. Default is same directory.
        - exclude (list or array-like, optional, default = None):
          contains names of events which are removed from events.
        - limits (tuple or list with two elements, optional,
          default = None): sets specific limits on y-axis.
        - scale (string, optional, default = None): if not None, it is
          given as argument to set_yscale().
        - save (boolean, optional, default = False): if True, the plot
          gets saved to path with a generated name.
        - path (string, optional, default = None): path of directory to
          save plot to. Default is same directory. If path contains a
          '.', like for example in 'pictures/photo.png', it is assumed
          to also contain a file name, so in the example a png-file
          named 'photo' will be saved to the subfolder 'pictures'.

    Returns:
        - Either a list containing a list with [mean of difference
          params, standard deviation of difference params, mean of
          difference PCs, standard deviation of difference PCs] for
          every GW event in events or None.
    """


    # Check conditions
    if basis and (basis != 'Phenom') and (basis != 'EOB'):
        raise ValueError('Basis has to be either \'Phenom\' or \'EOB\'.')

    if (model != 'nocosmo') and (model != 'cosmo'):
        raise ValueError('Model has to be either \'nocosmo\' or \'cosmo\'.')

    if exclude:
        for event in exclude:
            events.remove(event)
            
            
    # Store frequently used constants
    n = len(events)
    parlen = len(params)
    
    # Generate differences in parameters and PCs for each event
    if criterion == 'jsd2':
        # Special case because event_check has no option crit = 'jsd2'
        results = np.array(event_check(params, events, 'jsd', normed = True,
                                       centered = True, model = model,
                                       plot = False, threshold = threshold,
                                       directory = directory,
                                       exclude = exclude, limits = limits))
    else:
        results = np.array(event_check(params, events, criterion,
                                       normed = False, centered = centered,
                                       model = model, plot = False,
                                       threshold = threshold,
                                       directory = directory,
                                       exclude = exclude, limits = limits))
    

    if basis:
        results_PCs = np.array(event_check_PCs(basis, params, events,
                                               criterion, normed = normed,
                                               centered = centered,
                                               model = model, plot = False,
                                               threshold = threshold,
                                               directory = directory,
                                               exclude = exclude,
                                               limits = limits))


        # Return list with average criterion values or plot of them
        if plot:

            fig, ax = plt.subplots(figsize = (n / 2, 6))

            x = [i for i in range(n)]

            
            for k in range(parlen):
                ax.plot(x, results[:, k], 'o', color = 'b')
                ax.plot(x, results_PCs[:, k], 'o', color = 'r')
            
            
            ax.axline((0, threshold), (n - 1, threshold), color = 'r')

            ax.legend(labels = ['Original data', 'PCs' + basis],
                      bbox_to_anchor = (1.0, 0.5), loc = 'center left')
            # ax.set_title(title)
            ax.grid(axis = 'y')

            if scale is not None:
                ax.set_yscale(scale)
            
            # Set number of ticks for x-axis and label ticks
            ax.set_xticks(x, events, rotation = 45,
                          horizontalalignment = 'right',
                          rotation_mode = 'anchor')#rotation = 'vertical')

            if limits:
                ax.set_ylim(limits[0], limits[1])
            
            # Create pdf of plot
            if save:
                # If path already contains a name + data format, this name is
                # taken; otherwise, an automatic one is created
                if '.' in path[-4:]:
                    plt.savefig(path, bbox_inches = 'tight')
                else:
                    name = f'compare_quality_points_{basis}_{model}_crit'\
                           f'{criterion}_{str(params)}'
                    # if limits:
                    #     name += f'_limits{limits}'

                    if normed:
                        name += '_normed'
                    if centered:
                        name += 'centered'
                    if scale is not None:
                      name += scale
                    
                    plt.savefig(path + name + '.pdf', bbox_inches = 'tight')
        
        else:
            return [results, results_PCs]
    else:
        results_PCs_ph = np.array(event_check_PCs('Phenom', params, events,
                                                  criterion, normed = normed,
                                                  centered = centered,
                                                  model = model, plot = False,
                                                  threshold = threshold,
                                                  directory = directory,
                                                  exclude = exclude,
                                                  limits = limits))
        results_PCs_eob = np.array(event_check_PCs('EOB', params, events,
                                                   criterion, normed = normed,
                                                   centered = centered,
                                                   model = model, plot = False,
                                                   threshold = threshold,
                                                   directory = directory,
                                                   exclude = exclude,
                                                   limits = limits))


        # Return list with average criterion values or plot of them
        if plot:

            fig, ax = plt.subplots(figsize = (n / 2, 6))

            x = [i for i in range(n)]

            
            for k in range(parlen):
                ax.plot(x, results[:, k], 'o', color = 'b')
                ax.plot(x, results_PCs_ph[:, k], 'o', color = 'r')
                ax.plot(x, results_PCs_eob[:, k], 'o', color = 'g')
            
            
            ax.axline((0, threshold), (n - 1, threshold), color = 'r')

            ax.legend(labels = ['Original data', 'PCs Phenom', 'PCs EOB'],
                      bbox_to_anchor = (1.0, 0.5), loc = 'center left')
            # ax.set_title(title)
            ax.grid(axis = 'y')

            if scale is not None:
                ax.set_yscale(scale)
            
            # Set number of ticks for x-axis and label ticks
            ax.set_xticks(x, events, rotation = 45,
                          horizontalalignment = 'right',
                          rotation_mode = 'anchor')#rotation = 'vertical')

            if limits:
                ax.set_ylim(limits[0], limits[1])
            
            # Create pdf of plot
            if save:
                # If path already contains a name + data format, this name is
                # taken; otherwise, an automatic one is created
                if '.' in path[-4:]:
                    plt.savefig(path, bbox_inches = 'tight')
                else:
                    name = f'compare_quality_points_{model}_crit{criterion}'\
                           f'_{str(params)}'
                    # if limits:
                    #     name += f'_limits{limits}'

                    if normed:
                        name += '_normed'
                    if centered:
                        name += 'centered'
                    if scale is not None:
                      name += scale
                    
                    plt.savefig(path + name + '.pdf', bbox_inches = 'tight')
        
        else:
            return [results, results_PCs_ph, results_PCs_eob]






#
#
# add plt.show() after save in all functions which plot something!!!!
#
#
# documentation change: 'used' in argument basis at the end instead of 'put...'