import argparse
import os
import sys

from pathlib import Path
from tqdm import tqdm

import numpy as np
import yaml

from mne_bids import BIDSPath

# Hack-y way to import from files in sibling "io" directory
sys.path.append(str(Path(__file__).parent.parent / "io"))
sys.path.append(str(Path(__file__).parent.parent / "war_exp"))

from read import read_dataset, read_label, read_trial, get_trial_info
from plotting import (
    plot_signals,
    plot_roc_multiclass_cv,
    plot_feature_importances,
    plot_cv_indices,
)
from utils import NumpyEncoder
from cv import cv_fit


if __name__ == "__main__":
    # set time window
    # tmin, tmax = (-0.5, 1.0)
    # tmin, tmax = (-0.75, 1.25)  # This includes the Hit Target event, need to truncate
    # tmin, tmax = (-0.75, 0.5)   # Only encapsulates the Left Target event
    tmin, tmax = (-0.5, 0.5)

    bids_root = Path("/Volumes/Mac/research/data/efri/")

    derivatives_path = (
        bids_root
        / "derivatives"
        / "preprocessed"
        / f"tmin={tmin}-tmax={tmax}"
        / "band-pass=1-1000Hz-downsample=500"
    )

    if not os.path.exists(derivatives_path):
        os.makedirs(derivatives_path)

    # new directory paths for outputs and inputs at Hackerman workstation
    # bids_root = Path("/home/adam2392/hdd/Dropbox/efri/")
    # results_path = bids_root / "derivatives" / "raw" / "mtsmorf" / "results"

    ###### Some participants in the following list do not have MOVE data
    
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
            root=bids_root
        )

        bids_path = BIDSPath(**path_identifiers)

        # get EEG data
        picks = []
        epochs = read_dataset(
            bids_path,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            event_key="Left Target",
            notch_filter=True,
        )
        epochs.load_data()

        # Low-pass filter up to sfreq/2
        fs = epochs.info["sfreq"]
        epochs = epochs.filter(l_freq=1, h_freq=fs / 2 - 1)

        # Downsample epochs to 500 Hz
        resample_rate = 500
        epochs = epochs.resample(resample_rate)

        if not os.path.exists(derivatives_path / subject):
            os.makedirs(derivatives_path / subject)

        fname = os.path.splitext(bids_path.basename)[0] + "-epo.fif"
        fpath = derivatives_path / subject / fname
        epochs.save(fpath, overwrite=True)

        print(f"{subject.upper()} epochs saved at {fpath}")
