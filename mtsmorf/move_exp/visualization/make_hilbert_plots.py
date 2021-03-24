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
from mtsmorf.move_exp.plotting import plot_roc_multiclass_cv
from mtsmorf.move_exp.cv import cv_fit
from mtsmorf.io.utils import NumpyEncoder



if __name__ == "__main__":

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
    for subject in tqdm(subjects):
        destination = results_path / "decode_directionality" / subject / "hilbert_transform"
        if not os.path.exists(destination):
            os.makedirs(destination)

        epochs = read_move_trial_epochs(bids_root, subject, resample_rate=None)
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

        epochs_data = epochs.get_data()
        ntrials, nchs, nsteps = epochs_data.shape

        ntrials, nchs, nsteps = epochs_data.shape
        X = epochs_data.reshape(ntrials, -1)
        y = labels
        
        clf = rerfClassifier(
            projection_matrix="MT-MORF",
            max_features="auto",
            n_jobs=-1,
            random_state=rng,
            image_height=nchs,
            image_width=nsteps,
        )

        scores = cv_fit(clf, X, y, cv=cv, metrics=metrics, n_jobs=1, return_estimator=False)

        fig, ax = plt.subplots(dpi=100, figsize=(6.4, 4.8))

        plot_roc_multiclass_cv(
            scores["test_predict_proba"], labels, scores["test_inds"], label="", show_chance=True, ax=ax
        )
        ax.set_title("Hilbert Transformed Data")
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=False)         # ticks along the top edge are off
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            right=False)       # ticks along the top edge are off
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.tight_layout()
        plt.savefig(destination / f"{subject}_time_domain.png")
        with open(destination / f"{subject}_MT-MORF_time_domain.json", "w") as f:
                json.dump(scores, f, cls=NumpyEncoder)

        hilbert_data = {}
        for band, (l_freq, h_freq) in frequency_bands.items():
            epochs_band = epochs.copy().filter(l_freq=l_freq, h_freq=h_freq).apply_hilbert(envelope=True)
            epochs_band = epochs_band.resample(resample_rate)
            hilbert_data[band] = epochs_band.get_data()

        scores = {}
        for band, data in hilbert_data.items():
            ntrials, nchs, nsteps = data.shape
            X = data.reshape(ntrials, -1)
            y = labels
            
            clf = rerfClassifier(
                projection_matrix="MT-MORF",
                max_features="auto",
                n_jobs=-1,
                random_state=rng,
                image_height=nchs,
                image_width=nsteps,
            )
            scores[band] = cv_fit(clf, X, y, cv=cv, metrics=metrics, n_jobs=1, return_estimator=False)
            with open(destination / f"{subject}_MT-MORF_{band}.json", "w") as f:
                json.dump(scores[band], f, cls=NumpyEncoder)

        fig, axs = plt.subplots(2, 3, dpi=100, figsize=(6.4*3, 4.8*2))
        axs = axs.flatten()

        for ax, (band, score) in zip(axs, scores.items()):
            plot_roc_multiclass_cv(
                score["test_predict_proba"], labels, score["test_inds"], label="", show_chance=True, ax=ax
            )
            ax.set_title(f"{band} ({frequency_bands[band][0]}-{frequency_bands[band][1]} Hz)")
            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                top=False)         # ticks along the top edge are off
            ax.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                right=False)         # ticks along the top edge are off
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        fig.tight_layout()
        plt.savefig(destination / f"{subject}_all_frequency_bands_separate.png")

        data = np.stack(list(hilbert_data.values()), axis=2)

        ntrials, nchs, nfreqs, nsteps = data.shape
        X = data.reshape(ntrials, -1)
        y = labels

        clf = rerfClassifier(
            projection_matrix="MT-MORF",
            max_features="auto",
            n_jobs=-1,
            random_state=rng,
            image_height=nchs*nfreqs,
            image_width=nsteps,
        )
        scores_stacked = cv_fit(clf, X, y, cv=cv, metrics=metrics, n_jobs=1, return_estimator=True)

        fig, ax = plt.subplots(dpi=100, figsize=(6.4, 4.8))

        plot_roc_multiclass_cv(
            scores_stacked["test_predict_proba"], labels, scores_stacked["test_inds"], label="", show_chance=True, ax=ax
        )
        ax.set_title("Hilbert Transformed Data")
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=False)         # ticks along the top edge are off
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            right=False)       # ticks along the top edge are off
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.tight_layout()
        plt.savefig(destination / f"{subject}_all_frequency_bands_stacked.png")
        with open(destination / f"{subject}_MT-MORF_stacked.json", "w") as f:
                json.dump(scores_stacked, f, cls=NumpyEncoder)