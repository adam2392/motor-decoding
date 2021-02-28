import json
import os
import sys
import yaml
from pathlib import Path

import numpy as np
from mne_bids.path import BIDSPath
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split
from rerf.rerfClassifier import rerfClassifier

from functions.time_window_selection_functions import fit_classifiers_cv_time_window
from patch_selection import patch_selection, randomized_patch_selection

sys.path.append(str(Path(__file__).parent.parent / "io"))

from utils import NumpyEncoder


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
                             patch_width=5, patch_height=5, scoring="roc_auc",
                             random_state=1)

    result = randomized_patch_selection(clf, X_test, y_test, image_height, 
                                        image_width, patch_width=5, patch_height=5, 
                                        scoring="roc_auc", random_state=1)


def efri_movement_test():
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

    destination_path = results_path / subject / "patch_selection_experiment"
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Fit classifiers to trial-specific time windowed data
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

    result = patch_selection(best_clf, X_test, y_test, nchs, nsteps, patch_height=5, 
                             patch_width=10, scoring="roc_auc_ovr", n_repeats=5,
                             random_state=seed)

    scores["validate_roc_auc_ovr_imp_mean"] = result.importances_mean.tolist()
    scores["validate_roc_auc_ovr_imp_std"] = result.importances_std.tolist()
    scores["validate_patch_inds"] = result.patch_inds.tolist()

    # Re-save for MT-MORF cross-validation scores with patch selection results
    estimators = scores.get("estimator", None)
    del scores["estimator"]
    with open(destination_path / f"{subject}_MT-MORF_results.json", "w") as fout:
        json.dump(scores, fout, cls=NumpyEncoder)
        print(f"CV results for MT-MORF re-saved as json.")


if __name__ == "__main__":
    gaussian_classification_test()
    efri_movement_test()