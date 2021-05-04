import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rerf.rerfClassifier import rerfClassifier
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

if str(Path(__file__).parents[3]) not in sys.path:
    sys.path.append(str(Path(__file__).parents[3]))

from mtsmorf.io.move.read import read_move_trial_epochs, read_trial_metadata
from mtsmorf.move_exp.plotting import plot_feature_importances
from mtsmorf.move_exp.patch_selection import randomized_patch_selection
from mtsmorf.move_exp.cv import cv_fit
from mtsmorf.io.utils import NumpyEncoder


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="subject ID (e.g. efri02)")
    args = parser.parse_args()
    subject = args.subject

    with open(Path(__file__).parents[1] / "config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(Path(__file__).parents[1] / "metadata.yml") as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)

    bids_root = Path(config["bids_root"])
    results_path = Path(config["results_path"])
    subjects = metadata["subjects"]

    resample_rate = 500
    rng = 1
    n_splits = 5
    cv = StratifiedKFold(n_splits)
    metrics = dict(
        accuracy="accuracy",
        cohen_kappa_score=make_scorer(cohen_kappa_score),
        roc_auc_ovr="roc_auc_ovr",
    )

    tmin, tmax = (0, 0.25)
    destination = (
        results_path / "decode_directionality" / subject / "hilbert_transform" / f"tmin={tmin}_tmax={tmax}"
    )
    if not os.path.exists(destination):
        os.makedirs(destination)

    epochs = read_move_trial_epochs(bids_root, subject, tmin=tmin, tmax=tmax, resample_rate=None)
    epochs.load_data()
    trials_metadata = pd.DataFrame(read_trial_metadata(bids_root, subject))
    trials_metadata.head()

    stable_trials = trials_metadata.query("perturbed == False & success == True")
    stable_trials.head()
    labels = stable_trials.target_direction.values

    frequency_bands = dict(
        delta=(0.5, 4),
        theta=(4, 8),
        alpha=(8, 13),
        beta=(13, 30),
        gamma=(30, 70),
        hi_gamma=(70, 200),
    )

    epochs_resampled = epochs.filter(1, epochs.info["sfreq"] / 2. - 1).resample(resample_rate)
    epochs_data = epochs_resampled.get_data()

    ntrials, nchs, nsteps = epochs_data.shape
    X = epochs_data.reshape(ntrials, -1)
    y = labels

    old_results_path = Path(f"/Users/ChesterHuynh/OneDrive - Johns Hopkins/efri/derivatives/workstation_output/mtmorf/decode_directionality/{subject}/tmin=0_tmax=0.25_shuffle=True/time_domain")

    with open(old_results_path / f"{subject}_MT-MORF_results.json", "r") as f:
        scores = json.load(f)

    best = np.argmax(scores["test_roc_auc_ovr"])
    best_train_inds = scores["train_inds"][best]
    best_test_inds = scores["test_inds"][best]

    Xtest = X[best_test_inds]
    ytest = y[best_test_inds]

    best_clf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=-1,
        random_state=rng,
        image_height=nchs,
        image_width=nsteps,
    )
    best_clf.fit(X[best_train_inds], y[best_train_inds])

    result = randomized_patch_selection(
        best_clf,
        Xtest,
        ytest,
        nchs,
        nsteps,
        patch_height=5,
        patch_width=20,
        n_patches=2500,
        scoring="roc_auc_ovr",
        n_repeats=5,
        n_jobs=1,
        random_state=rng,
    )
    scores["validate_roc_auc_ovr_imp_mean"] = result.importances_mean.tolist()
    scores["validate_roc_auc_ovr_importances"] = result.importances.tolist()
    scores["validate_roc_auc_ovr_patch_inds"] = result.patch_inds.tolist()
    scores["validate_roc_auc_ovr_usage_counts"] = result.usage_counts.tolist()

    with open(destination / f"{subject}_MT-MORF_time_domain.json", "w") as f:
        json.dump(scores, f)

    # fig, ax = plt.subplots(dpi=100, figsize=(10, 16))
    # plot_feature_importances(
    #     result,
    #     epochs_resampled.ch_names,
    #     times=epochs_resampled.times,
    #     image_height=nchs,
    #     image_width=nsteps,
    #     ax=ax,
    # )
    # ax.set_title("Feature Importances (Time Domain)")
    # fig.tight_layout()
    # plt.savefig(destination / f"{subject}_patch_selection_time_domain.png")

    hilbert_data = {}
    for band, (l_freq, h_freq) in frequency_bands.items():
        epochs_band = (
            epochs.copy()
            .filter(l_freq=l_freq, h_freq=h_freq)
            .apply_hilbert(envelope=True)
        )
        epochs_band = epochs_band.resample(resample_rate)
        hilbert_data[band] = epochs_band.get_data()

    scores = {}
    for band, data in hilbert_data.items():
        ntrials, nchs, nsteps = data.shape
        X = data.reshape(ntrials, -1)
        y = labels
        
        old_results_path = Path(f"/Users/ChesterHuynh/OneDrive - Johns Hopkins/efri/derivatives/workstation_output/mtmorf/decode_directionality/{subject}/tmin=0_tmax=0.25_shuffle=True/time_domain/")
        with open(old_results_path / f"{subject}_MT-MORF_results_{band}.json", "r") as f:
            scores[band] = json.load(f)

        best = np.argmax(scores[band]["test_roc_auc_ovr"])
        best_train_inds = scores[band]["train_inds"][best]
        best_test_inds = scores[band]["test_inds"][best]
        Xtest = X[best_test_inds]
        ytest = y[best_test_inds]

        best_clf = rerfClassifier(
            projection_matrix="MT-MORF",
            max_features="auto",
            n_jobs=-1,
            random_state=rng,
            image_height=nchs,
            image_width=nsteps,
        )
        best_clf.fit(X[best_train_inds], y[best_train_inds])

        result = randomized_patch_selection(
            best_clf,
            Xtest,
            ytest,
            nchs,
            nsteps,
            patch_height=5,
            patch_width=20,
            n_patches=2500,
            scoring="roc_auc_ovr",
            n_repeats=5,
            n_jobs=1,
            random_state=rng,
        )

        scores[band]["validate_roc_auc_ovr_imp_mean"] = result.importances_mean.tolist()
        scores[band]["validate_roc_auc_ovr_importances"] = result.importances.tolist()
        scores[band]["validate_roc_auc_ovr_patch_inds"] = result.patch_inds.tolist()
        scores[band]["validate_roc_auc_ovr_usage_counts"] = result.usage_counts.tolist()
        with open(destination / f"{subject}_MT-MORF_{band}.json", "w") as f:
            json.dump(scores[band], f, cls=NumpyEncoder)

        # fig, ax = plt.subplots(dpi=100, figsize=(10, 16))
        # plot_feature_importances(
        #     result,
        #     epochs_resampled.ch_names,
        #     times=epochs_resampled.times,
        #     image_height=nchs,
        #     image_width=nsteps,
        #     ax=ax,
        # )
        # ax.set_title(f"Feature Importances ({band})")
        # fig.tight_layout()
        # plt.savefig(destination / f"{subject}_patch_selection_{band}.png")

    data = np.stack(list(hilbert_data.values()), axis=2)

    ntrials, nchs, nfreqs, nsteps = data.shape
    X = data.reshape(ntrials, -1)
    y = labels

    clf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=-1,
        random_state=rng,
        image_height=nchs * nfreqs,
        image_width=nsteps,
    )
    scores_stacked = cv_fit(
        clf, X, y, cv=cv, metrics=metrics, n_jobs=1, return_estimator=True
    )
    with open(destination / f"{subject}_MT-MORF_stacked.json", "r") as f:
        scores_stacked = json.load(f)

    best = np.argmax(scores_stacked["test_roc_auc_ovr"])
    best_train_inds = scores_stacked["train_inds"][best]
    best_test_inds = scores_stacked["test_inds"][best]
    Xtest = X[best_test_inds]
    ytest = y[best_test_inds]
    
    best_clf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=-1,
        random_state=rng,
        image_height=nchs*nfreqs,
        image_width=nsteps,
    )
    best_clf.fit(X[best_train_inds], y[best_train_inds])

    result = randomized_patch_selection(
        best_clf,
        Xtest,
        ytest,
        nchs*nfreqs,
        nsteps,
        patch_height=5,
        patch_width=20,
        n_patches=2500,
        scoring="roc_auc_ovr",
        n_repeats=5,
        n_jobs=1,
        random_state=rng,
    )

    scores_stacked["validate_roc_auc_ovr_imp_mean"] = result.importances_mean.tolist()
    scores_stacked["validate_roc_auc_ovr_importances"] = result.importances.tolist()
    scores_stacked["validate_roc_auc_ovr_patch_inds"] = result.patch_inds.tolist()
    scores_stacked["validate_roc_auc_ovr_usage_counts"] = result.usage_counts.tolist()
    with open(destination / f"{subject}_MT-MORF_stacked.json", "w") as f:
        json.dump(scores_stacked, f, cls=NumpyEncoder)

    ch_names = [f"{ch}-{band}" for ch in epochs.ch_names for band in frequency_bands]
    fig, ax = plt.subplots(dpi=100, figsize=(10, 16))
    plot_feature_importances(
        result,
        ch_names,
        times=epochs_resampled.times,
        image_height=nchs*nfreqs,
        image_width=nsteps,
        ax=ax,
    )
    ax.set_title(f"Feature Importances (All bands)")
    fig.tight_layout()
    plt.savefig(destination / f"{subject}_patch_selection_all_frequency_bands.png")
