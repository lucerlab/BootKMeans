''' K-means clustering with bootstrap uncertainty '''

# Author: Luis Cerdán <lcerdanphd@gmail.com>
# License: MIT 

import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin

class KMeansBoot(ClusterMixin, BaseEstimator):
    ''' 
    K-Means clustering with bootstrap to compute centroids
    and cluster labels uncertainties.    
    It is based on the sklearn implementation of K-Means.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Parameters
    -----------    
    n_boot: int, default = 999
        Number of bootstrap replicas to generate        
    n_clusters, init, n_init, max_iter, tol, verbose, random_state, copy_x, algorithm:
        The same as sklearn K-Means

    Attributes
    -----------    
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers (for full dataset).         
    labels_ : ndarray of shape (n_samples,)
        Labels of each point (for full dataset).
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided  (for full dataset).
    n_iter_ : int
        Number of iterations run  (for full dataset).
    n_features_in_ : int
        Number of features seen during :term:`fit`  (for full dataset).


    cluster_centers_boot_ : ndarray of shape (n_clusters, n_features, n_boot)
        Coordinates of cluster centers for each bootstrap replica model.
    labels_boot_ : ndarray of shape (n_samples, n_boot)
        Labels of each point for each bootstrap replica model.
    labels_boot_prob_ : ndarray of shape (n_samples, n_clusters)
        Probabilities of each point to belong to the different clusters. The probability
        is given by the number of times a point belongs to the given cluster divided by
        n_boot.
    inertia_boot_ : ndarray of shape (n_boot,)
        Sum of squared distances of samples to their closest cluster center for each
        bootstrap replica, weighted by the sample weights if provided.

    '''

    def __init__(
        self,
        n_boot=999,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="auto",
    ):

        self.n_boot = n_boot
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    def _check_params(self, X):
        # n_boot
        if self.n_boot <= 0:
            raise ValueError(f"n_boot should be > 0, got {self.n_init} instead.")
        self._n_boot = self.n_boot

    def fit(self, X, y=None, sample_weight=None):
        """Bootstraps K-Means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        self._check_params(X)

        # instantiate and fit Kmeans on full dataset
        self.kmeans_full = KMeans(n_clusters = self.n_clusters,
                                    init = self.init,
                                    max_iter = self.max_iter,
                                    tol = self.tol,
                                    n_init = self.n_init,
                                    verbose = self.verbose,
                                    random_state = self.random_state,
                                    copy_x = self.copy_x,
                                    algorithm = self.algorithm).fit(X)

        # KMeans Attributes
        self.cluster_centers_ = self.kmeans_full.cluster_centers_
        self.labels_ = self.kmeans_full.labels_
        self.inertia_ = self.kmeans_full.inertia_
        self.n_iter_ = self.kmeans_full.n_iter_
        self.n_features_in_ = self.kmeans_full.n_features_in_

        # Instantiate K-means model for boostrap replicas using self.cluster_centers_ as initial 
        # centroid guesses ('init'). Set n_init = 1 as 'init' provided
        kmeans_boot =  KMeans(n_clusters = self.n_clusters,
                                init = self.cluster_centers_,
                                max_iter = self.max_iter,
                                tol = self.tol,
                                n_init = 1,
                                verbose = self.verbose,
                                random_state = self.random_state,
                                copy_x = self.copy_x,
                                algorithm = self.algorithm
                                )

        # Bootstrap phase
        n_samples = X.shape[0]
        best_labels = np.empty((n_samples, self.n_boot))
        best_centers = np.empty((self.n_clusters, X.shape[1], self.n_boot))
        inertia_boot = np.empty(self.n_boot)

        ## Note: This loop could surely be parallelized with joblib
        for bb in range(self.n_boot):
            # sample with replacement
            Xb = X[np.random.randint(0, n_samples, n_samples),:]
            # find membership for each sample in X with model fitted to boostrap replica (Xb)
            best_labels[:,bb] = kmeans_boot.fit(Xb).predict(X)
            # get cluster centers and inertia
            best_centers[:,:,bb] = kmeans_boot.cluster_centers_
            inertia_boot[bb] = kmeans_boot.inertia_

        # compute membership probabilities        
        labels_boot_prob = [(best_labels == kk).sum(axis=1)/self.n_boot for kk in range(self.n_clusters)]

        # Bootstrapped KMeans Attributes
        self.cluster_centers_boot_ = best_centers
        self.labels_boot_ = best_labels
        self.labels_boot_prob_ = np.stack(labels_boot_prob, axis = 1)
        self.inertia_boot_ = inertia_boot

        return self    

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster for each sample in X (for full dataset model) and the
        probabilities to belong to each cluster.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        New data to predict.
        sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in X. If None, all observations
        are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        Index of the cluster each sample belongs to.
        labels_boot_prob: ndarray of shape (n_samples, n_clusters)
        Probabilities of each sample to belong to the different clusters.
        """

        check_is_fitted(self)

        # get predictions from KMeans fitted on full dataset (self.X)
        labels = self.kmeans_full.predict(X, sample_weight=sample_weight)

        # get distance of all new instances to bootstraped centroids
        l2 = ((X[:,None,:,None]-self.cluster_centers_boot_[None,:,:,:])**2).sum(axis = 2)

        # get indices of the centroids with the smallest distances
        labels_boot = np.argmin(l2, axis = 1)

        # compute membership probability
        labels_boot_prob = [(labels_boot == kk).sum(axis=1)/self.n_boot for kk in range(self.n_clusters)]

        return labels, np.stack(labels_boot_prob, axis = 1)