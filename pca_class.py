# -----------------------------------------------------------------------------
# 
# This script a class that can be used for principal component analysis
#
# Author: Max Melching
# Source: https://github.com/MaxMelching/bachelor_project
# 
# -----------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# This last import is from another script that was used for the thesis.
# If this is not available, removing jsd from this script is easy and
# does not change the functionality of it severely.
from criteria import jsd


def create_datamatrix(data: any, names: any = None) -> np.array:
    """
    Creates a matrix where each column stores repeated measurements in
    `data` of every parameter in `names`.

    Remark: this function was created to work with data from
    Gravitational-Wave posterior probability distributions published
    by the LIGO-Virgo-Kagra collaboration. However, it should also work
    with other data fulfilling the conditions described below.

    Parameters:
        - data (2D array-like): dataset where values of measurements are
          stored. Has to be of a type which allows access to subsets
          labeled with parameters par via data[par].
        - names (array-like, optional, default = None): set of
          parameters, order determines order of columns in output.
          If None, every parameter from data is taken.

    Returns:
        - numpy-array with columns equal to data[name] for each name in
          names.
    """
    
    if names is None:
        names = np.array([data.dtype.names]) # take every parameter
    else:
        names = np.array(names)
        
    return np.array([data[name] for name in names]).T # maybe = np.array(data[names]).T


class pca:
    def __init__(self, X: any, basis: any = None, names: any = None,
                 normed: bool = False, centered: bool = False):
        """
        Computes the Principal Components (PCs) of input data and
        provides additional functions to return (co-)variances,
        eigenvectors and more.

        Parameters:
        ----------
            - X (array-like): matrix containing dataset to work with.
              Columns represent different parameters, rows measurements.
            - basis (array-like, optional, default = None): data which
              is used to compute the transformation matrix (eigenvectors
              of the corresponding covariance matrix). Must have the
              same number of columns as X.
              If basis = None, it is set to X and the transformed X
              contains the Principal Components.
            - names (array-like, default = None): contains strings with
              labels of the parameters for the columns of X. Is only
              used in plots, so it can be None when no plots are used
              (otherwise, there will be an error).
            - normed (boolean, optional, default = False): if True, the
              data in X is normalized by dividing the values for
              each parameter by the respective standard deviation.
            - centered (boolean, optional, default = False): if True,
              the data in X is normalized by dividing the values for
              each parameter by the respective standard deviation.
        
        Attributes:
        ----------
            - X (numpy-array): copy of input X, which might be changed
              depending on the values of normed, centered.
            - basis (numpy-array): copy of input basis, which might be
              changed depending on the values of normed, centered. If
              the basis is None (default), it is set to X.
            - m (integer): number of rows in X.
            - n (integer): number of columns in X.
            - mb (integer): number of rows in basis.
            - nb (integer): number of columns in basis.
            - evals (numpy-array): eigenvalues of covariance matrix of
              basis (not necessarily X).
            - A (numpy-array): eigenvectors of covariance matrix of
              basis, arranged in a matrix.
            - Z (numpy-array): matrix product of X and A (transformed
              data). Equal to the PCs of X if basis = X.
            - Zbasis (numpy-array): matrix product of basis and A. Equal
              to Z if basis = X.
            - names (numpy-array): copy of input names.
        """
        
        # Create local variables
        self.X = np.array(X)
        
        if basis is None:
            self.basis = np.array(X)
        else:
            self.basis = np.array(basis)
        
        # Store dimensions
        self.m, self.n = self.X.shape
        self.mb, self.nb = self.basis.shape
        
        if centered:
            # Subtract row vector mean from every row of X. That does
            # not change variances of data/PCs, only shifts mean to zero
            self.X -= np.mean(self.X, axis=0)
            self.basis -= np.mean(self.basis, axis=0)
        
        
        if normed:
            # Divide each row by its standard deviation
            self.X /= np.std(self.basis, axis=0)

            self.basis /= np.std(self.basis, axis=0)
            
            self.S = np.corrcoef(self.basis, rowvar=False) 
            # np.cov would do the same because basis is standardized,
            # but this way we make sure and it is more obvious
        else:
            self.S = np.cov(self.basis, rowvar=False)
            # = 1 / (self.m - 1) * np.dot(self.basis.T, self.basis)
        
        # Diagonalise covariance matrix
        self.evals, self.A = np.linalg.eigh(self.S)

        # Sort eigenvalues and -vectors in descending order
        perm = np.flip(self.evals.argsort()) # argsort gives ascending
        self.evals = self.evals[perm]
        self.A = self.A[:, perm]
        
        # Compute transformed data
        self.Z = np.dot(self.X, self.A)

        self.Zbasis = np.dot(self.basis, self.A)
        
        
        self.names = np.array(names)
        # Permuting them like evals, A is not necessary as only
        # order of eigenvectors is (potentially) switched, but not the
        # components in them (which names correspond to).
        

    def get_X(self, q: int = None) -> np.array:
        """
        Returns the data matrix stored as a local variable. May differ
        from input X due to normalization and/or centering.

        Parameters:
        ----------
            - q (int, optional, default = None): number of columns to
              return. If None, all of them are taken.
        
        Returns:
        -------
            - first q columns of local variable X.
        """

        if q is None:
            q = self.n

        return self.X[:, :q]
        

    def get_basis(self, q: int = None) -> np.array:
        """
        Returns a certain number of columns of the basis matrix stored
        as a local variable. These may differ from input basis due to
        normalization and/ or centering.

        Parameters:
        ----------
            - q (int, optional, default = None): number of columns to
              return. If None, all of them are taken.
        
        Returns:
        -------
            - first q columns of local variable basis.
        """

        if q is None:
            q = self.n

        return self.basis[:, :q]
    

    def get_PCs(self, q: int = None) -> np.array:
        """
        Returns a certain number of columns of the transformed data
        matrix stored as a local variable, which contains the PCs of X
        if basis = X.

        Parameters:
        ----------
            - q (int, optional, default = None): number of columns to
              return. If None, all of them are taken.
        
        Returns:
        -------
            - first q columns of local variable Z.
        """

        if q is None:
            q = self.n

        return self.Z[:, :q] # = np.dot(X, A[:, :q])
    
    
    def get_basisPCs(self, q: int = None) -> np.array:
        """
        Returns a certain number of columns of the transformed basis
        matrix used in the function, which contains the PCs of basis.

        Parameters:
        ----------
            - q (int, optional, default = None): number of columns to
              return. If None, all of them are taken.
        
        Returns:
        -------
            - first q columns of local variable Zbasis.
        """
        
        if q is None:
            q = self.n

        return self.Zbasis[:, :q]
    

    def get_cov(self, q: int = None, display: bool = False,
                savepath: str = None) -> np.array | None:
        """
        Returns the covariance matrix of the first q columns of basis
        either as a 2D numpy-array or as a heatmap.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvalues
              to return. If None, all of them are taken.
            - display (boolean, optional, default = False): determines
              if the function returns a numpy-array (which can be used
              for storing the matrix) or if it is displayed.
            - savepath (string, optional, default = None): if display is
              True and savepath is not None, the plot generated will be
              saved. The filename is either given in saveplot (when a
              fileformat is contained in savepath) or an automatic one
              is created and saved to the path given by savepath (to
              save it into the same directory, set savepath = '').

        Returns:
        -------
            - first q columns of local variable S if display = True,
              else None.
        """
        
        if q is None:
            q = self.n

        if not display:
            return self.S[:q, :q]
        else:
            cov_df = pd.DataFrame(
                                  np.round(self.S[:q, :q], 2),
                                  columns=self.names[:q],
                                  index = self.names[:q]
                                  )
            sns.heatmap(cov_df.abs(), annot=cov_df, fmt='g')
            
            if savepath:
                # If path already contains a name + data format, this
                # name is taken. Otherwise, an automatic one is created.
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches='tight')
                else:
                    name = f'covmatrix_{self.names}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches='tight')
                    
            plt.show()
    

    def get_transformedcov(self, q: int = None, display: bool = False,
                           savepath: str = None) -> np.array | None:
        """
        Computes the diagonalized covariance matrix of the first q
        columns of basis, i.e. the matrix product (A^T)SA. Returned
        either as a 2D numpy-array or as a heatmap.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvalues
              to return. If None, all of them are taken.
            - display (boolean, optional, default = False): determines
              if the function returns a numpy-array (which can be used
              for storing the matrix) or if it is displayed.
            - savepath (string, optional, default = None): if display is
              True and savepath is not None, the plot generated will be
              saved. The filename is either given in saveplot (when a
              fileformat is contained in savepath) or an automatic one
              is created and saved to the path given by savepath (to
              save it into the same directory, set savepath = '').

        Returns:
        -------
            - first q columns of first q columns of diagonalized
              covariance matrix of basis if display = True, else None.
        """
        
        if q is None:
            q = self.n

        Sa = np.dot(self.A.T, np.dot(self.S, self.A))
        
        if not display:
            return Sa
        else:
            cov_df = pd.DataFrame(
                                  np.round(Sa[:q, :q], 2),
                                  columns=self.names[:q],
                                  index=self.names[:q]
                                  )
            sns.heatmap(cov_df.abs(), annot=cov_df, fmt='g')
            # Limits 0 and 1 could be chosen for normed = True
            
            if savepath:
                # If path already contains a name + data format, this
                # name is taken. Otherwise, an automatic one is created.
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches='tight')
                else:
                    name = f'transformed_covariancematrix{self.names}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches='tight')
                    
            plt.show()
    

    def get_eigenvalues(self, q: int = None) -> np.array | None:
        """
        Returns a certain number of eigenvalues of the covariance matrix
        of basis in descending order.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvalues
              to return. If None, all of them are taken.
        
        Returns:
        -------
            - first q columns of local variable evals.
        """

        if q is None:
            q = self.n
        
        return self.evals[:q] # = np.var(self.Zbasis[:, :q], axis=0)
    

    def get_transformedvariances(self, q: int = None, display: bool = False,
                                 savepath: str = None) -> np.array | None:
        """
        Returns a certain number of variances of parameters in Z, Zbasis.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvalues
              to return. If None, all of them are taken.
            - display (boolean, optional, default = False): determines
              if the function returns a numpy-array (which can be used
              for storing the matrix) or if it is displayed.
            - savepath (string, optional, default = None): if display is
              True and savepath is not None, the plot generated will be
              saved. The filename is either given in saveplot (when a
              fileformat is contained in savepath) or an automatic one
              is created and saved to the path given by savepath (to
              save it into the same directory, set savepath = '').
        
        Returns:
        -------
            - matrix of first q variances of Z, Zbasis in each row if
              display = True, else None.
        """
        
        if q is None:
            q = self.n
        
        zvar = np.var(self.Z[:, :q], axis=0)

        if not display:
            return [zvar, self.evals]
        else:
            pcconvert1 = 1 / np.sum(self.evals) * 100 # conversion factor; will be used multiple times, thus define here
            pcconvert2 = 1 / np.sum(zvar) * 100 # not necessarily equal to pcconvert1 since X, basis may have different total variance
            percents = [f'{i + 1} ({self.evals[i] * pcconvert1: .2f}% vs. {zvar[i] * pcconvert2: .2f}% of variance)' for i in range(q)]        
            df = pd.DataFrame(np.array([np.round(zvar, 2), np.round(self.evals[:q], 2)]).T, columns = ['Variances of Z', 'Variances of Zbasis'], index = percents)
            sns.heatmap(df, vmin = 0, annot = True, fmt = 'g')
            #plt.xlabel('number') # kind of unnecessary? Should be clear
            plt.ylabel('Components')

            if savepath:
                # If path already contains a name + data format, this
                # name is taken. Otherwise, an automatic one is created.
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches='tight')
                else:
                    name = f'transformed_variances_{self.names}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches='tight')
                    
            plt.show()
    

    def get_retainedvariance(self, q: int = None) -> float:
        """
        Calculates the percentage of variance retained by a certain
        number of eigenvalues.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvalues
              to take. If None, all of them are taken.
        
        Returns:
        -------
            - sum of the first q eigenvalues divided by 100 to give a
              percentage.
        """

        if q is None:
            q = self.n
        
        return np.sum(self.evals[:q]) / np.sum(self.evals) * 100
    

    def get_eigenvectors(self, q: int = None, display: bool = False,
                         savepath: str = None) -> np.array | None:
        """
        Returns a certain number of eigenvectors of S.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvectors
              to return. If None, all of them are taken.
            - display (boolean, optional, default = False): determines
              if the function returns a numpy-array (which can be used
              for storing the matrix) or if it is displayed.
            - savepath (string, optional, default = None): if display is
              True and savepath is not None, the plot generated will be
              saved. The filename is either given in saveplot (when a
              fileformat is contained in savepath) or an automatic one
              is created and saved to the path given by savepath (to
              save it into the same directory, set savepath = '').

        Returns:
        -------
            - matrix of first q eigenvectors of basis if display = True,
              else None.
        """

        if q is None:
            q = self.n
        
        # shows components with respect to basis of parameters, so the first component
        # of the first eigenvector is the 'contribution' of parameter 1
        if not display:
            return self.A[:, :q]
        else:
            pcconvert = 1 / np.sum(self.evals) * 100 # conversion factor; will be used multiple times, thus defined here
            percents = [f'{i + 1} ({self.evals[i] * pcconvert: .2f}% of variance)' for i in range(q)]
            eigenvec_df = pd.DataFrame(np.round(self.A[:, :q].T, 2), columns = self.names, index = percents)
            # no .T and change columns, index? -> nah, looks weird with components written downwards
            
            sns.heatmap(eigenvec_df.abs(), vmin = 0, vmax = 1, annot = eigenvec_df, fmt = 'g', linewidths = 0.5)
            # cmap is based on color palette rocket from seaborn (default)
            # like this, color is based on abs and annot writes real values including minus signs into the cells
            
            plt.title(f'The first {q} eigenvalues account for {np.sum(self.evals[0:q]) / np.sum(self.evals) * 100: .2f}% of the variance')
            plt.xlabel('Parameters')
            plt.ylabel('Eigenvectors')
            
            # maybe divide each vector (= row) by its maximum element? Then, relative influence can be seen better, right?

            if savepath:
                # If path already contains a name + data format, this
                # name is taken. Otherwise, an automatic one is created.
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches='tight')
                else:
                    name = f'eigenvectors_{self.names}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches='tight')
                    
            plt.show()


    def get_weightedeigenvectors(self, q: int = None, display: bool = False,
                                 savepath: str = None) -> np.array | None:
        """
        Computes the weighted versions of eigenvectors of S where
        weights are the square root of the respective eigenvalue. This
        visualizes the significance of the eigenvector in a convenient
        manner and is also called correlation.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvectors
              to return. If None, all of them are taken.
            - display (boolean, optional, default = False): determines
              if the function returns a numpy-array (which can be used
              for storing the matrix) or if it is displayed.
            - savepath (string, optional, default = None): if display is
              True and savepath is not None, the plot generated will be
              saved. The filename is either given in saveplot (when a
              fileformat is contained in savepath) or an automatic one
              is created and saved to the path given by savepath (to
              save it into the same directory, set savepath = '').

        Returns:
        -------
            - matrix of first q weighted eigenvectors of basis if
              display = True, else None.
        """

        #corr = self.A * np.sqrt(self.evals) / np.std(self.X, axis = 0) # std is 1 for normed = True
        #corr = self.A * np.sqrt(np.var(self.Z, axis = 0)) / np.std(self.X, axis = 0) 
        # although not strict from definition -> divide by std(Z)??? Will be 1 for our case, so doesn't matter for now

        corr = self.A[:, :q] * np.sqrt(self.evals[:q])

        if not display:
            return corr
        else:        
            PCs = [f'PC {i}' for i in range(1, self.n + 1)]
            corr_df = pd.DataFrame(np.round(corr.T, 2), columns = self.names, index = PCs)
            
            sns.heatmap(corr_df.abs(), vmin = 0, annot = corr_df, fmt = 'g', linewidths=0.5)
            plt.title('Correlation between parameters and PCs')
            plt.xlabel('Parameters')
            plt.ylabel('PCs')

            if savepath:
                # If path already contains a name + data format, this
                # name is taken. Otherwise, an automatic one is created.
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches='tight')
                else:
                    name = f'weighted_eigenvectors_{self.names}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches='tight')
                    
            plt.show()
        

    def get_eigenvectors_standarderror(self, q: int = None, display = False,
                                       savepath: str = None) -> np.array | None:
        """
        Calculates the error for each component of a certain amount of
        eigenvectors of S based on a formula taken from "A User's Guide
        to Principal Component Analysis" by J.E. Jackson.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvectors
              to return. If None, all of them are taken.
            - display (boolean, optional, default = False): determines
              if the function returns a numpy-array (which can be used
              for storing the matrix) or if it is displayed.
            - savepath (string, optional, default = None): if display is
              True and savepath is not None, the plot generated will be
              saved. The filename is either given in saveplot (when a
              fileformat is contained in savepath) or an automatic one
              is created and saved to the path given by savepath (to
              save it into the same directory, set savepath = '').

        Returns:
        -------
            - matrix of the error of the first q eigenvectors of basis
              if display = True, else None.
        """

        if not q:
            q = self.n

        m = self.basis.shape[0]
        # it is correct to take the eigenvalues here as they are obtained from self.basis (which is what we want)
        errors = np.sqrt([[self.evals[i] / m * np.sum([self.evals[h] / (self.evals[h] - self.evals[i])**2 * self.A[g, h]**2 for h in range(self.n) if h != i]) for i in range(self.n)] for g in range(self.n)])
        #np.sqrt(self.evals / self.m * np.sum(self.evals * self.A, axis = 0))

        if not display:
            return errors[:, :q]
        else:        
            pcconvert = 1 / np.sum(self.evals) * 100 # conversion factor; will be used multiple times, thus define here
            percents = [f'{i + 1} ({self.evals[i] * pcconvert: .2f}% of variance)' for i in range(q)]
            eigenvec_df = pd.DataFrame(np.round(errors[:, :q].T, 2), columns = self.names, index = percents)
            # no .T and change columns, index? -> nah, looks weird with components written downwards
            
            # maybe divide each vector (= row) by its maximum element? Then, relative influence can be seen better, right?
            
            sns.heatmap(eigenvec_df.abs(), vmin = 0, vmax = 1, annot = eigenvec_df, fmt = 'g', linewidths = 0.5)
            # cmap is based on color palette rocket from seaborn (default)
            # like this, color is based on abs and annot writes real values including minus signs into the cells
            
            #plt.title(f'The first {q} eigenvalues account for {np.sum(self.evals[0:q]) / np.sum(self.evals) * 100: .2f}% of the variance')
            plt.title(f'Standard error of the first {q} eigenvectors')
            plt.xlabel('Parameters')
            plt.ylabel('Eigenvectors')

            if savepath:
                # If path already contains a name + data format, this
                # name is taken. Otherwise, an automatic one is created.
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches='tight')
                else:
                    name = f'eigenvectors_standarderror_{self.names}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches='tight')
                    
            plt.show()
                
            
    def get_transformedmeans(self, q: int = None, display: bool = False,
                             savepath: str = None) -> np.array | None:
        """
        Returns a certain number of means of parameters in Z, Zbasis.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvalues
              to return. If None, all of them are taken.
            - display (boolean, optional, default = False): determines
              if the function returns a numpy-array (which can be used
              for storing the matrix) or if it is displayed.
            - savepath (string, optional, default = None): if display is
              True and savepath is not None, the plot generated will be
              saved. The filename is either given in saveplot (when a
              fileformat is contained in savepath) or an automatic one
              is created and saved to the path given by savepath (to
              save it into the same directory, set savepath = '').
        
        Returns:
        -------
            - matrix of first q variances of Z, Zbasis in each row if
              display = True, else None.
        """

        if q is None:
            q = self.n

        means = [np.mean(self.Z[:, :q], axis=0), np.mean(self.Zbasis[:, :q], axis=0)]

        if not display:
            return means
        else:
            df = pd.DataFrame(np.round(means, 2).T, columns = ['Means of Z', 'Means of Zbasis'], index = [f'{i + 1}' for i in range(q)])
            sns.heatmap(df.abs(), annot = df, fmt = 'g')
            plt.ylabel('Components')

            if savepath:
                # If path already contains a name + data format, this
                # name is taken. Otherwise, an automatic one is created.
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches='tight')
                else:
                    name = f'transformed_means_{self.names}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches='tight')
                    
            plt.show()
            
            
    def get_originalmeans(self, q: int = None, display: bool = False,
                          savepath: str = None) -> np.array | None:
        """
        Returns a certain number of means of parameters in X, basis.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvalues
              to return. If None, all of them are taken.
            - display (boolean, optional, default = False): determines
              if the function returns a numpy-array (which can be used
              for storing the matrix) or if it is displayed.
            - savepath (string, optional, default = None): if display is
              True and savepath is not None, the plot generated will be
              saved. The filename is either given in saveplot (when a
              fileformat is contained in savepath) or an automatic one
              is created and saved to the path given by savepath (to
              save it into the same directory, set savepath = '').
        
        Returns:
        -------
            - matrix of first q variances of X, basis in each row if
              display = True, else None.
        """

        if q is None:
            q = self.n

        means = [np.mean(self.X[:, :q], axis=0), np.mean(self.basis[:, :q], axis=0)]

        if not display:
            return means
        else:
            df = pd.DataFrame(np.round(means, 2).T, columns = ['Means of X', 'Means of basis'], index = self.names)
            sns.heatmap(df.abs(), annot = df, fmt = 'g')
            plt.ylabel('Parameters')

            if savepath:
                # If path already contains a name + data format, this
                # name is taken. Otherwise, an automatic one is created.
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches='tight')
                else:
                    name = f'original_variances_{self.names}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches='tight')
                    
            plt.show()
    

    def get_originalvariances(self, q: int = None, display: bool = False,
                              savepath: str = None) -> np.array | None:
        """
        Returns a certain number of variances of parameters in X, basis.

        Parameters:
        ----------
            - q (int, optional, default = None): number of eigenvalues
              to return. If None, all of them are taken.
            - display (boolean, optional, default = False): determines
              if the function returns a numpy-array (which can be used
              for storing the matrix) or if it is displayed.
            - savepath (string, optional, default = None): if display is
              True and savepath is not None, the plot generated will be
              saved. The filename is either given in saveplot (when a
              fileformat is contained in savepath) or an automatic one
              is created and saved to the path given by savepath (to
              save it into the same directory, set savepath = '').
        
        Returns:
        -------
            - matrix of first q variances of X, basis in each row if
              display = True, else None.
        """
        
        if q is None:
            q = self.n
        
        xvar = np.var(self.X[:, :q], axis = 0)
        basisvar = np.var(self.basis[:, :q], axis = 0)

        if not display:
            return [xvar, basisvar]
        else:
            pcconvert1 = 1 / np.sum(basisvar) * 100 # conversion factor; will be used multiple times, thus define here
            pcconvert2 = 1 / np.sum(xvar) * 100 # not necessarily equal to pcconvert1 since X, basis may have different total variance
            percents = [f'{self.names[i]} ({basisvar[i] * pcconvert1: .2f}% vs. {xvar[i] * pcconvert2: .2f}% of variance)' for i in range(q)]        
            df = pd.DataFrame(np.array([np.round(xvar, 2), np.round(basisvar[:q], 2)]).T, columns = ['Variances of X', 'Variances of basis'], index = percents)
            sns.heatmap(df, vmin = 0, annot = True, fmt = 'g')
            #plt.xlabel('number') # kind of unnecessary? Should be clear
            plt.ylabel('Parameters')

            if savepath:
                # If path already contains a name + data format, this
                # name is taken. Otherwise, an automatic one is created.
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches='tight')
                else:
                    name = f'original_variances_{self.names}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches='tight')
                    
            plt.show()
    

    # next: some functions which are specifically for PCA, mainly for selecting relevant PCs
    def total_variance(self, t: float) -> int:
        """
        Computes the number of PCs necessary to keep a certain
        percentage of the total variance for the transformed basis.

        Parameters:
        ----------
            - t (float): threshold of percentage to be kept. Has to be
              0 <= t <= 1.
        
        Returns:
        -------
            - q (integer): minimal number of variables to keep the
              percentage t of the total variance.
        """

        retained_var = 0
        q = 1
        total_var = np.sum(self.evals)
        
        for i in range(self.n):
            retained_var += self.evals[i] / total_var
            if retained_var >= t:
                break
            else:
                q += 1
        
        return q


    def scree_plot(self, q: int = None, savepath: str = None) -> None:
        """
        Creates a scree plot for the variances of a certain amount of
        the transformed variables.

        Parameters:
        ----------
            - q (int, optional, default = None): number of variables
              to return. If None, all of them are taken.
            - savepath (string, optional, default = None): if it is not
              None, the plot generated will be saved. The filename is
              either given in saveplot (when a fileformat is contained
              in savepath) or an automatic one is created and saved to
              the path given by savepath (to save it into the same
              directory, set savepath = '').
        
        Returns:
        -------
            - None
        """

        if q is None:
            q = self.n

        x = np.arange(1, self.n + 1, 1)
        
        plt.plot(x, self.evals, marker='x', markersize=10, markeredgecolor='r')
        #plt.bar(x, height = self.evals, width = 0.5)
        plt.ylabel('Eigenvalue')
        plt.xlabel('Component')
        plt.ylim(0, 1.2 * self.evals[0])
        plt.xticks(x)
        plt.grid(True)
        for i, eval in enumerate(self.evals): # indices and values are needed
            plt.text(i + 1, 1.05 * eval + 0.05, np.round(eval, 2), ha='center') # for bar: second to '1.05 * eval'
            # +1 because i starts at 0 instead of 1; ha = horizontalalignment

        if savepath:
            # If path already contains a name + data format, this name
            # is taken. Otherwise, an automatic one is created.
            if '.' in savepath[-4:]:
                plt.savefig(savepath, bbox_inches='tight')
            else:
                name = f'screeplot_{q}PCs_{self.names}'
                # if limits:
                #     name += f'_limits{limits}'
                
                plt.savefig(savepath + name + '.pdf', bbox_inches='tight')

        plt.show()

        # the following ones are showing both variances, but is that really what we want?
        # Criteria are specifically about PCs...

        # x = np.arange(1, q + 1, 1)
        # zvar = np.array(np.var(self.Z[:, :q], axis = 0))
        
        # plt.plot(x, self.evals[:q], marker = 'x', markersize = 10, markeredgecolor = 'r', label = 'eigenvalue')
        # plt.plot(x, zvar, marker = 'x', markersize = 10, markeredgecolor = 'g', label = 'variance')
        # plt.ylabel('value')
        # plt.xlabel('variable number')
        # plt.ylim(0, 1.2 * self.evals[0])
        # plt.xticks(x)
        # plt.grid(True)
        # plt.legend(loc = 'upper right')
        # for i, eval in enumerate(self.evals[:q]): # indices and values are needed
        #     plt.text(i + 1, eval + 0.1, np.round(eval, 2), ha = 'center') # for bar: second to '1.05 * eval'
        #     # +1 because i starts at 0 instead of 1; ha = horizontalalignment
        # plt.show()

        # x = np.arange(1, q + 1, 1)
        # zvar = np.array(np.var(self.Z[:, :q], axis = 0))

        # fig, axs = plt.subplots(ncols = 2, sharex = True, sharey = True, figsize = (16, 5))
        
        # axs[0].plot(x, zvar, marker = 'x', markersize = 10, markeredgecolor = 'g')
        # axs[1].plot(x, self.evals[:q], marker = 'x', markersize = 10, markeredgecolor = 'r')
        # axs[0].set_ylabel('Variance')
        # axs[1].set_ylabel('Variance')
        # axs[0].set_xlabel('Variable number')
        # axs[1].set_xlabel('Variable number')

        # axs[0].set_ylim(0, 1.2 * max(zvar[0], self.evals[0]))
        # axs[0].set_xticks(x)
        # # axs[0].grid(True)
        # # axs[1].grid(True)
        # plt.grid(True)
        
        # for i, var in enumerate(zvar[:q]): # indices and values are needed
        #     axs[0].text(i + 1, var + 0.1, np.round(var, 2), ha = 'center') # for bar: second to '1.05 * eval'
        #     # +1 because i starts at 0 instead of 1; ha = horizontalalignment
        
        # for i, eval in enumerate(self.evals[:q]): # indices and values are needed
        #     axs[1].text(i + 1, eval + 0.1, np.round(eval, 2), ha = 'center') # for bar: second to '1.05 * eval'
        #     # +1 because i starts at 0 instead of 1; ha = horizontalalignment

        # plt.show()
    
        
    def kaiser(self) -> int:
        """
        Computes the number of relevant PCs based on Kaiser's rule.

        Parameters:
        ----------
            - None
        
        Returns:
        -------
            - q (integer): number of PCs with variance greater than 0.7.
              This is in accordance with what I.T. Jolliffe argues in
              his book "Principal Component Analysis".
        """

        eigenvecs = self.A.T[np.where(self.evals > 0.7)]
        q = eigenvecs.shape[0] # [1] would also be ok, is quadratic
        return q
    

    def broken_stick(self) -> np.array:
        """
        Computes which PCs should be kept based on the broken stick
        model approach.

        Parameters:
        ----------
            - None
        
        Returns:
        -------
            - indices (numpy-array): mask for array-likes showing which
              PCs should be kept. Can be used as A[:, indices] to get
              the "important" eigenvectors.
        """

        #eigenvecs = np.array([self.A[:, i] for i in range(self.n) if (self.evals[i] > np.sum([1 / j for j in range(i, self.n + 1)]) / self.n)]).T
        indices = np.array([i - 1 for i in range(1, self.n + 1) if (self.evals[i - 1] >= np.sum([1 / j for j in range(i, self.n + 1)]) / self.n)])

        return indices
    

    # does not work yet, therefore commented
    # def t2_check(self, t): # t = threshold, given as percentile
    #     Znorm = self.Z # / self.evals
    #     t2vals = np.sum(Znorm**2, axis = 1) # sums through rows, gives T^2-value for every measurement
    #     t2threshold = self.n * (self.m - 1) / (self.m - self.n) * f.ppf(t, self.n, self.m - self.n)
    #     # formula from Jackson, equation 1.7.5; ppf directly gives the point, when 95% of pdf are reached
    #     t2filtered = t2vals[np.where(t2vals <= t2threshold)]
        
    #     return [t2threshold, t2filtered.size, t2vals.size] # needs adjustment for normed case, way too manyy are discarded
    #                                                        # -> not dividing by self.evals is better, but does that make sense?
    #                                                        # -> I don't think so, because although X has normalised variance, Z doesn't (eigenvalues != 1)
    

    # lastly: functions to plot parts of the data
    def compare_plot(self, k: int, savepath: str = None) -> None:
        """
        Compares the histograms of the k-th variable from X and basis.

        Parameters:
        ----------
            - k (integer): index of variable (column) to take.
            - savepath (string, optional, default = None): if it is not
              None, the plot generated will be saved. The filename is
              either given in saveplot (when a fileformat is contained
              in savepath) or an automatic one is created and saved to
              the path given by savepath (to save it into the same
              directory, set savepath = '').
        
        Returns:
        -------
            - None
        """

        Xk = self.X[:, k]
        basisk = self.basis[:, k]
        
        # fig, axs = plt.subplots(ncols = 2, sharex = True, sharey = True, figsize = (16, 5))
        # #plt.suptitle(f'PC{k} contains {np.var(Zk) / np.sum(self.evals) * 100: .2f}% of the total variance\n'
        # #         + f'PC{k} contains {self.evals[k] / np.sum(self.evals) * 100: .2f}% of the total variance\n')
        # plt.suptitle(f'Parameter: {self.names[k]}, $D_{{JS}} = ${jsd(Xk, basisk)}')
        # axs[0].set_title('X')
        # axs[1].set_title('Basis')
        
        # # plot data points
        # axs[0].hist(Xk, bins = int((self.m * 4) ** (1 / 3)), density = True)
        # axs[1].hist(basisk, bins = int((basisk.size * 4) ** (1 / 3)), density = True)
        
        # axs[0].grid(True)
        # axs[1].grid(True)
        
        # plt.show()


        plt.figure(figsize = (8, 5))
        plt.title(f'{self.names[k]},\t$D_{{JS}} = ${jsd(Xk, basisk): .4f}\n'
                  + f'{np.var(Xk) / np.sum(np.var(self.Z, axis = 0)) * 100: .2f}%,'
                  + f'{self.evals[k] / np.sum(self.evals) * 100: .2f}% of total variance for X, basis')

        plt.hist(Xk, bins = int((self.m * 4) ** (1 / 3)), density = True, histtype = 'step', label = 'X')
        plt.hist(basisk, bins = int((self.mb * 4) ** (1 / 3)), density = True, histtype = 'step', label = 'basis')

        plt.grid(True)
        plt.legend()

        if savepath:
            # if path already contains a name + data format, this name is
            # taken; otherwise, an automatic one is created
            if '.' in savepath[-4:]:
                plt.savefig(savepath, bbox_inches = 'tight')
            else:
                name = f'compare_plot_param{self.names[k]}'
                # if limits:
                #     name += f'_limits{limits}'
                
                plt.savefig(savepath + name + '.pdf', bbox_inches = 'tight')
                
        plt.show()
    

    def compare_plot_PCs(self, k: int, savepath: str = None) -> None:
        """
        Compares the histograms of the k-th transformed variable from
        Z and Zbasis.

        Parameters:
        ----------
            - k (integer): index of PC (column) to take.
            - savepath (string, optional, default = None): if it is not
              None, the plot generated will be saved. The filename is
              either given in saveplot (when a fileformat is contained
              in savepath) or an automatic one is created and saved to
              the path given by savepath (to save it into the same
              directory, set savepath = '').
        
        Returns:
        -------
            - None
        """

        Zk = self.Z[:, k]
        Zbasisk = self.Zbasis[:, k]
        
        # fig, axs = plt.subplots(ncols = 2, sharex = True, sharey = True, figsize = (16, 5))
        # #plt.suptitle(f'PC{k} contains {np.var(Zk) / np.sum(self.evals) * 100: .2f}% of the total variance\n'
        # #         + f'PC{k} contains {self.evals[k] / np.sum(self.evals) * 100: .2f}% of the total variance\n')
        # plt.suptitle(f'PC{k + 1}, $D_{{JS}} = ${jsd(Zk, Zbasisk): .4f}')
        # # axs[0].set_title(f'PC{k + 1} contains {np.var(Zk) / np.sum(self.evals) * 100: .2f}% of the total variance and has mean {np.mean(Zk): .2f}')
        # # axs[1].set_title(f'PC{k + 1} contains {self.evals[k] / np.sum(self.evals) * 100: .2f}% of the total variance and has mean {np.mean(Zbasisk): .2f}')
        # axs[0].set_title(f'{np.var(Zk) / np.sum(np.var(self.Z, axis = 0)) * 100: .2f}% of total variance')
        # axs[1].set_title(f'{self.evals[k] / np.sum(self.evals) * 100: .2f}% of total variance')
                
        # # plot data points
        # axs[0].hist(Zk, bins = int((self.m * 4) ** (1 / 3)), density = True)
        # axs[1].hist(Zbasisk, bins = int((Zbasisk.size * 4) ** (1 / 3)), density = True)
        
        # axs[0].grid(True)
        # axs[1].grid(True)
        
        # plt.show()


        plt.figure(figsize = (8, 5))
        plt.title(f'Component {k + 1},\t$D_{{JS}} = ${jsd(Zk, Zbasisk): .4f}\n'
                  + f'{np.var(Zk) / np.sum(np.var(self.Z, axis = 0)) * 100: .2f}%,'
                  + f'{self.evals[k] / np.sum(self.evals) * 100: .2f}% of total variance for Z, Zbasis')

        plt.hist(Zk, bins = int((self.m * 4) ** (1 / 3)), density = True, histtype = 'step', label = 'Z')
        plt.hist(Zbasisk, bins = int((self.mb * 4) ** (1 / 3)), density = True, histtype = 'step', label = 'Zbasis')

        plt.grid(True)
        plt.legend()

        if savepath:
            # if path already contains a name + data format, this name is
            # taken; otherwise, an automatic one is created
            if '.' in savepath[-4:]:
                plt.savefig(savepath, bbox_inches = 'tight')
            else:
                name = f'compare_plot_PC{k}_{self.names}'
                # if limits:
                #     name += f'_limits{limits}'
                
                plt.savefig(savepath + name + '.pdf', bbox_inches = 'tight')
                
        plt.show()
    

    def compare_plot_all(self, savepath: str = None) -> None:
        """
        Plots histograms of every original and transformed variable.

        Parameters:
        ----------
            - savepath (string, optional, default = None): if it is not
              None, the plot generated will be saved. The filename is
              either given in saveplot (when a fileformat is contained
              in savepath) or an automatic one is created and saved to
              the path given by savepath (to save it into the same
              directory, set savepath = '').
        
        Returns:
        -------
            - None
        """

        # fig, axs = plt.subplots(ncols = 2, nrows = self.n, figsize = (16, 5 * self.n))
        fig, axs = plt.subplots(ncols = self.n, nrows = 2, figsize = (8 * self.n, 10))
        plt.suptitle('Comparison of 1D-projections')

        xbins = int((self.m * 4) ** (1 / 3))
        basisbins = int((self.mb * 4) ** (1 / 3))

        # for i in range(self.n):
        #     axs[i, 0].hist(self.X[:, i], bins = xbins, density = True, histtype = 'step', label = 'X')
        #     axs[i, 0].hist(self.basis[:, i], bins = basisbins, density = True, histtype = 'step', label = 'Basis')
        #     axs[i, 0].set_title(f'{self.names[i]},  $D_{{JS}} = ${jsd(self.X[:, i], self.basis[:, i]): .4f}')
        #     axs[i, 0].legend()
        #     axs[i, 0].grid(True)

        #     axs[i, 1].hist(self.Z[:, i], bins = xbins, density = True, histtype = 'step', label = 'Z')
        #     axs[i, 1].hist(self.Zbasis[:, i], bins = basisbins, density = True, histtype = 'step', label = 'Zbasis')
        #     axs[i, 1].set_title(f'Component {i + 1},  $D_{{JS}} = ${jsd(self.Z[:, i], self.Zbasis[:, i]): .4f}')
        #     axs[i, 1].legend()
        #     axs[i, 1].grid(True)

        for i in range(self.n):
            axs[0, i].hist(self.X[:, i], bins = xbins, density = True, histtype = 'step', label = 'X')
            axs[0, i].hist(self.basis[:, i], bins = basisbins, density = True, histtype = 'step', label = 'Basis')
            axs[0, i].set_title(f'{self.names[i]},  $D_{{JS}} = ${jsd(self.X[:, i], self.basis[:, i]): .4f}')
            axs[0, i].legend()
            axs[0, i].grid(True)

            axs[1, i].hist(self.Z[:, i], bins = xbins, density = True, histtype = 'step', label = 'Z')
            axs[1, i].hist(self.Zbasis[:, i], bins = basisbins, density = True, histtype = 'step', label = 'Zbasis')
            axs[1, i].set_title(f'Component {i + 1},  $D_{{JS}} = ${jsd(self.Z[:, i], self.Zbasis[:, i]): .4f}')
            axs[1, i].legend()
            axs[1, i].grid(True)

        if savepath:# is not None:
            # if path already contains a name + data format, this name is
            # taken; otherwise, an automatic one is created
            if '.' in savepath[-4:]:
                plt.savefig(savepath, bbox_inches = 'tight')
            else:
                name = f'compare_plot_all_{self.names}'
                # if limits:
                #     name += f'_limits{limits}'
                
                plt.savefig(savepath + name + '.pdf', bbox_inches = 'tight')
        
        plt.show()
    

    def compare_plot_2D(self, i: int, j: int, separate: bool = False,
                        savepath: str = None) -> None:
        """
        Plots the projection of X and basis onto the plane of two
        variables.

        Parameters:
        ----------
            - i (integer): index of first variable to take.
            - j (integer): index of second variable to take.
            - separate (boolean, optional, default = False): determines
              if datasets are plotted next to each other or in one plot.
            - savepath (string, optional, default = None): if it is not
              None, the plot generated will be saved. The filename is
              either given in saveplot (when a fileformat is contained
              in savepath) or an automatic one is created and saved to
              the path given by savepath (to save it into the same
              directory, set savepath = '').
        
        Returns:
        -------
            - None
        """

        x1, y1 = self.X[:, i], self.X[:, j]
        x2, y2 = self.basis[:, i], self.basis[:, j]

        if separate:
            fig, axs = plt.subplots(ncols = 2, figsize = (16, 5))

            plt.suptitle('Comparison of 2D-projection')
            axs[0].set_title('X')
            axs[1].set_title('Basis')
            
            # plot data points
            axs[0].plot(x1, y1, 'x')
            axs[1].plot(x2, y2, 'x', color = 'orange')

            # hmmm, does following part make sense?
            # # get means, vectors will be drawn from there (otherwise range might be unnecessarily high)
            # center1 = np.mean([x1, y1], axis = 0)
            # center2 = np.mean([x2, y2], axis = 0)
            
            # # plot first two eigenvectors
            # axs[0].plot([center1[0], center1[0] + self.A[0, 0]], [center1[1], center1[1] + self.A[1, 0]], color = 'red', label = 'eigenvectors')
            # axs[0].plot([center1[0], center1[0] + self.A[0, 1]], [center1[1], center1[1] + self.A[1, 1]], color = 'red')
            # # idea: scale the eigenvectors with their respective eigenvalue? To also visualize how much variance contained?
            
            # # plot transformed eigenvectors
            # axs[1].plot([center2[0], center2[0] + self.A[0, 0]], [center2[1], center2[1] + self.A[1, 0]], color = 'red', label = 'eigenvectors')
            # axs[1].plot([center2[0], center2[0] + self.A[0, 1]], [center2[1], center2[1] + self.A[1, 1]], color = 'red')
            
            axs[0].grid(True)
            axs[1].grid(True)
            axs[0].set_aspect('equal')
            axs[1].set_aspect('equal')
            axs[0].set_xlabel(self.names[i])
            axs[0].set_ylabel(self.names[j])
            axs[1].set_xlabel(self.names[i])
            axs[1].set_ylabel(self.names[j])

            if savepath:
                # if path already contains a name + data format, this name is
                # taken; otherwise, an automatic one is created
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches = 'tight')
                else:
                    name = f'compare_plot_params{self.names[i]},{self.names[j]}_separate'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches = 'tight')

            plt.show()
        else:
            fig, ax = plt.subplots(figsize = (8, 5))

            plt.title('Comparison of 2D-projection')
            
            # plot data points
            ax.plot(x1, y1, 'x', label = 'X')
            ax.plot(x2, y2, 'x', label = 'Xbasis')

            ax.grid(True)
            ax.set_aspect('equal')
            ax.set_xlabel(self.names[i])
            ax.set_ylabel(self.names[j])

            if savepath:
                # if path already contains a name + data format, this name is
                # taken; otherwise, an automatic one is created
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches = 'tight')
                else:
                    name = f'compare_plot_params{self.names[i]},{self.names[j]}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches = 'tight')

            plt.show()
    

    def compare_plot_PCs_2D(self, i: int, j: int, separate = False,
                            savepath: str = None) -> None:
        """
        Plots the projection of Z and Zbasis onto the plane of two
        variables.

        Parameters:
        ----------
            - i (integer): index of first PC to take.
            - j (integer): index of second PC to take.
            - separate (boolean, optional, default = False): determines
              if datasets are plotted next to each other or in one plot.
            - savepath (string, optional, default = None): if it is not
              None, the plot generated will be saved. The filename is
              either given in saveplot (when a fileformat is contained
              in savepath) or an automatic one is created and saved to
              the path given by savepath (to save it into the same
              directory, set savepath = '').
        
        Returns:
        -------
            - None
        """

        x1, y1 = self.Z[:, i], self.Z[:, j]
        x2, y2 = self.Zbasis[:, i], self.Zbasis[:, j]
        
        if separate:
            fig, axs = plt.subplots(ncols = 2, figsize = (16, 5))#, sharex = True, sharey = True
            # plt.suptitle(f'eigenvector1 = {np.round(self.A[:, 0], 2)} contains {self.evals[0] / np.sum(self.evals) * 100: .2f}% of the total variance\n'
            #          + f'eigenvector2 = {np.round(self.A[:, 1], 2)} contains {self.evals[1] / np.sum(self.evals) * 100: .2f}% of the total variance\n')
            
            plt.suptitle('Comparison of 2D-projection')
            axs[0].set_title('Z')
            axs[1].set_title('Zbasis')

            # plot data points
            axs[0].plot(x1, y1, 'x')
            axs[1].plot(x2, y2, 'x', color = 'orange')

            # hmmm, does following part make sense?
            # # get means, vectors will be drawn from there (otherwise range might be unnecessarily high)
            # center1 = np.mean([x1, y1], axis = 0)
            # center2 = np.mean([x2, y2], axis = 0)
            
            # Atransformed = np.array(np.dot(np.array(self.A[:, i], self.A[:, j]).T, self.A)[:, i],
            # np.dot(np.array(self.A[:, i], self.A[:, j]).T, self.A)[:, j]) # will be coordinate axes
            
            # # plot first two eigenvectors
            # axs[0].plot([center1[0], center1[0] + Atransformed[0, 0]], [center1[1], center1[1] + Atransformed[1, 0]], color = 'red', label = 'eigenvectors')
            # axs[0].plot([center1[0], center1[0] + Atransformed[0, 1]], [center1[1], center1[1] + Atransformed[1, 1]], color = 'red')
            # # idea: scale the eigenvectors with their respective eigenvalue? To also visualize how much variance contained?
            
            # # plot transformed eigenvectors
            # axs[1].plot([center2[0], center2[0] + Atransformed[0, 0]], [center2[1], center2[1] + Atransformed[1, 0]], color = 'red', label = 'eigenvectors')
            # axs[1].plot([center2[0], center2[0] + Atransformed[0, 1]], [center2[1], center2[1] + Atransformed[1, 1]], color = 'red')
            
            axs[0].grid(True)
            axs[1].grid(True)
            axs[0].set_aspect('equal')
            axs[1].set_aspect('equal')
            axs[0].set_xlabel(f'PC {i + 1}')
            axs[0].set_ylabel(f'PC {j + 1}')
            axs[1].set_xlabel(f'PC {i + 1}')
            axs[1].set_ylabel(f'PC {j + 1}')

            if savepath:
                # if path already contains a name + data format, this name is
                # taken; otherwise, an automatic one is created
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches = 'tight')
                else:
                    name = f'compare_plot_PCs{i},{j}_separate'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches = 'tight')

            plt.show()
        else:
            fig, ax = plt.subplots(figsize = (8, 5))#, sharex = True, sharey = True
            # plt.suptitle(f'eigenvector1 = {np.round(self.A[:, 0], 2)} contains {self.evals[0] / np.sum(self.evals) * 100: .2f}% of the total variance\n'
            #          + f'eigenvector2 = {np.round(self.A[:, 1], 2)} contains {self.evals[1] / np.sum(self.evals) * 100: .2f}% of the total variance\n')
            
            plt.title('Comparison of 2D-projection')

            # plot data points
            ax.plot(x1, y1, 'x', label = 'Z')
            ax.plot(x2, y2, 'x', label = 'Zbasis')
            
            ax.grid(True)
            ax.set_aspect('equal')
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')

            if savepath:
                # if path already contains a name + data format, this name is
                # taken; otherwise, an automatic one is created
                if '.' in savepath[-4:]:
                    plt.savefig(savepath, bbox_inches = 'tight')
                else:
                    name = f'compare_plot_PCs{i},{j}'
                    # if limits:
                    #     name += f'_limits{limits}'
                    
                    plt.savefig(savepath + name + '.pdf', bbox_inches = 'tight')

            plt.show()