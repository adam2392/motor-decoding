import argparse
import os
import sys

from pathlib import Path
from tqdm import tqdm

import numpy as np
import yaml

from mne_bids import BIDSPath

from cv import cv_fit
from experiment_functions import preprocess_epochs
from plotting import (
    plot_signals,
    plot_roc_multiclass_cv,
    plot_feature_importances,
    plot_cv_indices,
)

# Hack-y way to import from files in sibling "io" directory
sys.path.append(str(Path(__file__).parent.parent / "io"))
from read import read_dataset, read_label, read_trial, get_trial_info
from utils import NumpyEncoder


if __name__ == "__main__":
    # set time window
    # tmin, tmax = (-0.5, 1.0)
    # tmin, tmax = (-0.75, 1.25)  # This includes the Hit Target event, need to truncate
    # tmin, tmax = (-0.75, 0.5)   # Only encapsulates the Left Target event
    tmin, tmax = (-0.5, 0.5)

    # configure paths from ./config.yml
    with open(Path(os.path.dirname(__file__)) / "config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    bids_root = Path(config["bids_root"])
    derivatives_path = (
        Path(config["derivatives_path"])
        / "preprocessed"
        / f"tmin={tmin}-tmax={tmax}"
        / "band-pass=1-1000Hz-downsample=500"
    )

    if not os.path.exists(derivatives_path):
        os.makedirs(derivatives_path)

    with open(Path(os.path.dirname(__file__)) / "metadata.yml") as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)

    subjects = metadata["subjects"]

    for subject in tqdm(subjects):

        # subject identifiers
        path_identifiers = dict(
            subject=subject,
            session="efri",
            task="move",
            acquisition="seeg",
            run="01",
            suffix="ieeg",
            extension=".vhdr",
            root=bids_root,
        )

        bids_path = BIDSPath(**path_identifiers)

        # get EEG data
        epochs = read_dataset(
            bids_path,
            tmin=tmin,
            tmax=tmax,
            event_key="Left Target",
            notch_filter=True,
        )
        epochs.load_data()
        epochs = preprocess_epochs(epochs)

        if not os.path.exists(derivatives_path / subject):
            os.makedirs(derivatives_path / subject)

        fname = os.path.splitext(bids_path.basename)[0] + "-epo.fif"
        fpath = derivatives_path / subject / fname
        epochs.save(fpath, overwrite=True)

        print(f"{subject.upper()} epochs saved at {fpath}.")
