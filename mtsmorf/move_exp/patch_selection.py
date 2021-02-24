import numpy as np
import pandas as pd
from rerf.rerfClassifier import rerfClassifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch, check_random_state, check_array


def _weights_scorer(scorer, estimator, X, y, sample_weight):
    if sample_weight is not None:
        return scorer(estimator, X, y, sample_weight)
    return scorer(estimator, X, y)


def _compute_vectorized_index(row_idx, col_idx, image_width):
    """Convert image indices to vectorized indices."""
    vectorized_idx = row_idx * image_width + col_idx
    return vectorized_idx


def _calculate_permutation_scores(estimator, X, y, sample_weight, patch_idx,
                                  random_state, n_repeats, scorer):
    """Calculate score when `col_idx` is permuted."""
    random_state = check_random_state(random_state)

    # Work on a copy of X to to ensure thread-safety in case of threading based
    # parallelism. Furthermore, making a copy is also useful when the joblib
    # backend is 'loky' (default) or the old 'multiprocessing': in those cases,
    # if X is large it will be automatically be backed by a readonly memory map
    # (memmap). X.copy() on the other hand is always guaranteed to return a
    # writable data-structure whose columns can be shuffled inplace.
    X_permuted = X.copy()
    scores = np.zeros(n_repeats)
    shuffling_idx = np.arange(X.shape[0])
    for n_round in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_permuted, "iloc"):
            patch = X_permuted.iloc[shuffling_idx, patch_idx]
            patch.index = X_permuted.index
            X_permuted.iloc[:, patch_idx] = patch
        else:
            X_permuted[:, patch_idx] = X_permuted[shuffling_idx][:, patch_idx]
        feature_score = _weights_scorer(
            scorer, estimator, X_permuted, y, sample_weight
        )
        scores[n_round] = feature_score

    return scores


def patch_selection(estimator, X, y, image_height, image_width, 
                    patch_width=5, *, scoring=None, n_repeats=5, n_jobs=None, 
                    random_state=None, sample_weight=None):
    """Based on Permutation importance for feature evaluation [BRE]_.
    The :term:`estimator` is required to be a fitted estimator. `X` can be the
    data set used to train the estimator or a hold-out set. The permutation
    importance of a feature is calculated as follows. First, a baseline metric,
    defined by :term:`scoring`, is evaluated on a (potentially different)
    dataset defined by the `X`. Next, a feature column from the validation set
    is permuted and the metric is evaluated again. The permutation importance
    is defined to be the difference between the baseline metric and metric from
    permutating the feature column.
    Parameters
    ----------
    estimator : object
        An estimator that has already been :term:`fitted` and is compatible
        with :term:`scorer`.
    X : ndarray or DataFrame, shape (n_samples, n_features)
        Data on which permutation importance will be computed.
    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
        Targets for supervised or `None` for unsupervised.
    image_height : int
        Height of unflattened sample.
    image_width : int
        Width of unflattened sample.
    patch_width : int, default=20
        Width of each patch in the patch selection.
    scoring : string, callable or None, default=None
        Scorer to use. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used.
    n_repeats : int, default=5
        Number of times to permute a feature.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel. The computation is done by computing
        permutation score for each columns and parallelized over the columns.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    random_state : int, RandomState instance, default=None
        Pseudo-random number generator to control the permutations of each
        feature.
        Pass an int to get reproducible results across function calls.
        See :term: `Glossary <random_state>`.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights used in scoring.
        .. versionadded:: 0.24
    Returns
    -------
    result : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        importances_mean : ndarray, shape (n_features, )
            Mean of feature importance over `n_repeats`.
        importances_std : ndarray, shape (n_features, )
            Standard deviation over `n_repeats`.
        importances : ndarray, shape (n_features, n_repeats)
            Raw permutation importance scores.
    References
    ----------
    .. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
             2001. https://doi.org/10.1023/A:1010933404324
    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.inspection import permutation_importance
    >>> X = [[1, 9, 9],[1, 9, 9],[1, 9, 9],
    ...      [0, 9, 9],[0, 9, 9],[0, 9, 9]]
    >>> y = [1, 1, 1, 0, 0, 0]
    >>> clf = LogisticRegression().fit(X, y)
    >>> result = permutation_importance(clf, X, y, n_repeats=10,
    ...                                 random_state=0)
    >>> result.importances_mean
    array([0.4666..., 0.       , 0.       ])
    >>> result.importances_std
    array([0.2211..., 0.       , 0.       ])
    """
    if not hasattr(X, "iloc"):
        X = check_array(X, force_all_finite='allow-nan', dtype=None)

    # Precompute random seed from the random state to be used
    # to get a fresh independent RandomState instance for each
    # parallel call to _calculate_permutation_scores, irrespective of
    # the fact that variables are shared or not depending on the active
    # joblib backend (sequential, thread-based or process-based).
    random_state = check_random_state(random_state)
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

    scorer = check_scoring(estimator, scoring=scoring)
    baseline_score = _weights_scorer(scorer, estimator, X, y, sample_weight)

    # List out the starting row and starting col indices for each patch
    # Patches are non-random, non-overlapping, and of size patch_width x patch_width.
    patch_row_idx = np.arange(0, image_height, patch_width)
    patch_col_idx = np.arange(0, image_width, patch_width)

    patches = [
        [
            _compute_vectorized_index(r, c, image_width)
            for r in np.arange(row_start, min(image_height, row_start + patch_width))
            for c in np.arange(col_start, min(image_width, col_start + patch_width))
        ] 
        for row_start in patch_row_idx
        for col_start in patch_col_idx
    ]

    scores = [
        _calculate_permutation_scores(
            estimator, X, y, sample_weight, patch, random_seed, n_repeats, scorer
        ) 
        for patch in patches
    ]

    importances = baseline_score - np.array(scores)
    return Bunch(importances_mean=np.mean(importances, axis=1),
                 importances_std=np.std(importances, axis=1),
                 importances=importances)


if __name__ == "__main__":
    np.random.seed(1)

    n = 50
    image_height, image_width = 10, 20
    class0 = np.random.randn(n // 2, image_height, image_width)
    class1 = np.random.randn(n // 2, image_height, image_width) + 2e-1

    data = np.vstack([class0, class1])
    labels = np.vstack([np.zeros((n // 2, 1)), np.ones((n // 2, 1))])

    shuffle_idx = np.random.choice(n, size=n, replace=False)
    X = data[shuffle_idx].reshape(n, -1)
    y = np.squeeze(labels[shuffle_idx])
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = rerfClassifier(projection_matrix="MT-MORF", n_jobs=-1, random_state=1,
                         image_height=image_height, image_width=image_width)
    clf.fit(X_train, y_train)

    result = patch_selection(clf, X_test, y_test, image_height, image_width, 
                             patch_width=5, scoring="roc_auc", n_jobs=-1, 
                             random_state=1)

    print("done")