import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from rerf.rerfClassifier import rerfClassifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch, check_random_state, check_array
import json
import os
import sys
import yaml
from pathlib import Path

from mne_bids.path import BIDSPath
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split
from rerf.rerfClassifier import rerfClassifier

from functions.time_window_selection_functions import fit_classifiers_cv_time_window

sys.path.append(str(Path(__file__).parent.parent / "io"))
from utils import NumpyEncoder


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


def patch_selection(estimator, X, y, image_height, image_width, *, patch_height=5, 
                    patch_width=5, scoring=None, n_repeats=5, n_jobs=None, 
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
    patch_height : int, default=5
        Width of each patch in the patch selection.
    patch_width : int, default=5
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
        importances_mean : ndarray, shape (n_patches, )
            Mean of feature importance over `n_repeats`.
        importances_std : ndarray, shape (n_patches, )
            Standard deviation over `n_repeats`.
        importances : ndarray, shape (n_patches, n_repeats)
            Raw permutation importance scores.
        patch_inds : ndarray, shape (n_patches, patch_height, patch_width)
            Indices for all patches used for importances.
    References
    ----------
    .. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
             2001. https://doi.org/10.1023/A:1010933404324
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
    # Patches are non-random, non-overlapping, and of size patch_height x patch_width.
    patch_row_idx = np.arange(0, image_height, patch_height)
    patch_col_idx = np.arange(0, image_width, patch_width)

    patches = np.array([
        np.array([
            _compute_vectorized_index(r, c, image_width)
            for r in np.arange(row_start, min(image_height, row_start + patch_height))
            for c in np.arange(col_start, min(image_width, col_start + patch_width))
        ])
        for row_start in patch_row_idx
        for col_start in patch_col_idx
    ])

    scores = Parallel(n_jobs=n_jobs)(delayed(_calculate_permutation_scores)(
            estimator, X, y, sample_weight, patch, random_seed, n_repeats, scorer
        ) for patch in patches)

    importances = baseline_score - np.array(scores)
    return Bunch(importances_mean=np.mean(importances, axis=1),
                 importances_std=np.std(importances, axis=1),
                 importances=importances,
                 patch_inds=patches)


def randomized_patch_selection(estimator, X, y, image_height, image_width, *, 
                               patch_height=5, patch_width=5, n_patches=None, 
                               scoring=None, n_repeats=5, n_jobs=None, 
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
    patch_height : int, default=5
        Width of each patch in the patch selection.
    patch_width : int, default=5
        Width of each patch in the patch selection.
    n_patches: int, or None, default=None
        Number of patches to compute for each repeat.
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
        importances_mean : ndarray, shape (n_patches, )
            Mean of feature importance over `n_repeats`.
        importances_std : ndarray, shape (n_patches, )
            Standard deviation over `n_repeats`.
        importances : ndarray, shape (n_patches, n_repeats)
            Raw permutation importance scores.
        patch_inds : ndarray, shape (n_patches, patch_height, patch_width)
            Indices for all patches used for importances.
    References
    ----------
    .. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
             2001. https://doi.org/10.1023/A:1010933404324
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

    if n_patches is None:
        # TODO: See what would be a good number to do.
        n_samples, n_features = X.shape
        n_patches = n_features // (patch_height * patch_width)

    # We construct patches as that are discontiguous along rows and contiguous
    # along columns. Stacked together, these patches have dimension of 
    # patch_height x patch_width.

    # Shuffle all row indices and select the first patch_height row indices
    patch_rows = [
        random_state.permutation(image_height)[:patch_height]
        for _ in range(n_patches)
    ]

    # Get a random column and the next patch_width sequential columns
    patch_cols = [
        random_state.choice(image_width - patch_width) + np.arange(patch_width)
        for _ in range(n_patches)
    ]
    
    # Create a meshgrid for all indices
    patches = np.array([
        np.array([
            _compute_vectorized_index(r, c, image_width)
            for r in rows
            for c in cols
        ])
        for rows, cols in zip(patch_rows, patch_cols)
    ])

    scores = Parallel(n_jobs=n_jobs)(delayed(_calculate_permutation_scores)(
            estimator, X, y, sample_weight, patch, random_seed, n_repeats, scorer
        ) for patch in patches)

    importances = baseline_score - np.array(scores)

    # Count number of times each pixel was used in patch sampling
    unique, counts = np.unique(patches, return_counts=True)
    
    usage_counts = np.zeros(image_height * image_width, dtype=np.int)
    usage_counts[unique] = counts
    
    # Sum all the importances for each pixel
    total_importances = np.zeros(image_height * image_width)
    summed_importances = np.repeat(np.sum(importances, axis=1), patches.shape[1])
    np.add.at(total_importances, patches.flatten(), summed_importances)

    # Average importance for each pixel based on how many times it was used
    mean_importances = np.zeros_like(total_importances)
    mean_importances = np.divide(total_importances, usage_counts, out=mean_importances, 
                                 where=(usage_counts != 0))

    # Reshape into a 2-D array
    mean_importances = mean_importances.reshape(image_height, image_width)
    return Bunch(importances_mean=mean_importances,
                #  importances_std=np.std(importances, axis=1),  # Figure out by append importances for each pixel later
                 importances=importances,
                 patch_inds=patches,
                 usage_counts=usage_counts)


def gaussian_classification_test():
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
                             patch_height=5, patch_width=5, scoring="roc_auc",
                             random_state=1)


def gaussian_classification_randomized_test():
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

    n_repeats = 5
    n_patches = 50
    patch_height = 5
    patch_width = 5
    result = randomized_patch_selection(clf, X_test, y_test, image_height, 
                                        image_width, n_repeats=n_repeats, 
                                        n_patches=n_patches, patch_height=patch_height, 
                                        patch_width=patch_width, scoring="roc_auc", 
                                        random_state=1)

    assert result.patch_inds.shape == (n_patches, patch_height * patch_width)
    assert result.importances.shape == (n_patches, n_repeats)


def efri_movement_test(save_results=False):
    """
    Target direction classification problem with EFRI07.
    Selecting trial-specific time windows and zeroing out any data that are not
    between the Go Cue and Hit Target events.
    Data is in the time domain.
    """

    with open(Path(os.path.dirname(__file__)) / "config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    subject = "efri07"
    bids_root = Path(config["bids_root"])
    results_path = Path(config["results_path"])
    path_identifiers = dict(subject=subject, session="efri", task="move", 
                            acquisition="seeg", run="01", suffix="ieeg",
                            extension=".vhdr", root=bids_root)
    bids_path = BIDSPath(**path_identifiers)

    seed = 1
    n_splits = 5
    cv = StratifiedKFold(n_splits)
    metrics = dict(
        accuracy="accuracy",
        cohen_kappa_score=make_scorer(cohen_kappa_score),
        roc_auc_ovr="roc_auc_ovr",
    )
    patch_height, patch_width = 5, 10

    destination_path = (
        results_path 
        / subject 
        / "patch_selection_experiment" 
        / f"{patch_height}x{patch_width}_patches"
    )
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Fit classifiers to trial-specific windowed data
    clf_scores, masked_data, labels = fit_classifiers_cv_time_window(
        bids_path, cv, metrics, "trial_specific", return_data=True, random_state=seed
    )

    # Save cross-validation scores
    for clf_name, scores in clf_scores.items():
        estimators = scores.get("estimator", None)
        del scores["estimator"]
        
        with open(destination_path / f"{subject}_{clf_name}_results.json", "w") as fout:
            json.dump(scores, fout, cls=NumpyEncoder)
            print(f"CV results for {clf_name} saved as json.")
        
        if estimators is not None:
            scores["estimator"] = estimators

    # Run patch selection on best classifier
    scores = clf_scores["MT-MORF"]
    best_split = np.argmax(scores["test_roc_auc_ovr"])
    best_clf = scores["estimator"][best_split]
    best_test_inds = scores["test_inds"][best_split]
    
    ntrials, nchs, nsteps = masked_data.shape
    X_test = masked_data.reshape(ntrials, -1)[best_test_inds]
    y_test = labels[best_test_inds]

    clf = rerfClassifier(projection_matrix="MT-MORF", n_jobs=-1, random_state=1,
                         image_height=nchs, image_width=nsteps)
    clf.fit(X_test, y_test)

    result = patch_selection(best_clf, X_test, y_test, nchs, nsteps, patch_height=patch_height, 
                             patch_width=patch_width, scoring="roc_auc_ovr", n_repeats=5,
                             random_state=seed)

    scores["validate_roc_auc_ovr_imp_mean"] = result.importances_mean.tolist()
    scores["validate_roc_auc_ovr_imp_std"] = result.importances_std.tolist()
    scores["validate_patch_inds"] = result.patch_inds.tolist()

    # Re-save for MT-MORF cross-validation scores with patch selection results
    if save_results:
        estimators = scores.get("estimator", None)
        del scores["estimator"]
        with open(destination_path / f"{subject}_MT-MORF_results.json", "w") as fout:
            json.dump(scores, fout, cls=NumpyEncoder)
            print(f"CV results for MT-MORF re-saved as json.")


if __name__ == "__main__":
    # gaussian_classification_test()
    gaussian_classification_randomized_test()
    # efri_movement_test()