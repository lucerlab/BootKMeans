# BootKMeans

## Overview

Sklearn type class that uses bootstrap resampling onto K Means clustering to obtain an estimate of the uncertainties in the cluster membership, centroids, and boundaries. 
It can be combined with pipelines to preprocess the training data. 

## Dependencies

BootKMeans requires sci-kit learn 1.0 and all its dependencies

## Installation

* Clone this repository to your local machine

```
$ git clone https://github.com/lucerlab/BootKMeans
```

* Do you want to make it `pip`-installable? Go ahead and feel free to contribute.

## Parameters and attributes

**Parameters**   
    * n_boot: int, default = 999
       - Number of bootstrap replicas to generate        
    * n_clusters, init, n_init, max_iter, tol, verbose, random_state, copy_x, algorithm:
       - The same as [sklearn K-Means] (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

**Attributes**   
    * cluster_centers_ : ndarray of shape (n_clusters, n_features)
      - Coordinates of cluster centers (for full dataset).         
    * labels_ : ndarray of shape (n_samples,)
      - Labels of each point (for full dataset).
    * inertia_ : float
      - Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided  (for full dataset).
    * n_iter_ : int
      - Number of iterations run  (for full dataset).
    * n_features_in_ : int
      - Number of features seen during :term:`fit`  (for full dataset).


    * cluster_centers_boot_ : ndarray of shape (n_clusters, n_features, n_boot)
      - Coordinates of cluster centers for each bootstrap replica model.
    * labels_boot_ : ndarray of shape (n_samples, n_boot)
      - Labels of each point for each bootstrap replica model.
    * labels_boot_prob_ : ndarray of shape (n_samples, n_clusters)
      - Probabilities of each point to belong to the different clusters. The probability
        is given by the number of times a point belongs to the given cluster divided by
        n_boot.
    * inertia_boot_ : ndarray of shape (n_boot,)
      - Sum of squared distances of samples to their closest cluster center for each
        bootstrap replica, weighted by the sample weights if provided.