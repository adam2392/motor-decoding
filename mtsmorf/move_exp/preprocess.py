import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    # bids_root = Path("/workspaces/research/mnt/data/efri/")
    bids_root = Path("/Volumes/Mac/research/data/efri/")
    derivatives_path = (
        bids_root / "derivatives" / "preprocessed" / "tmin=-0.75-tmax=1.25" / "low-pass=1000Hz-downsample=500"
    )
    # derivatives_path = (
    #     bids_root / "derivatives" / "preprocessed" / "band-pass=70-200Hz-downsample=500"
    # )

    if not os.path.exists(derivatives_path):
        os.makedirs(derivatives_path)

    # new directory paths for outputs and inputs at Hackerman workstation
    # bids_root = Path("/home/adam2392/hdd/Dropbox/efri/")
    # results_path = bids_root / "derivatives" / "raw" / "mtsmorf" / "results"

    ###### Some participants in the following list do not have MOVE data
    subjects = [
        "efri02",
        "efri06",
        "efri07",
        "efri09",  # Too few samples
        "efri10",  # Unequal data size vs label size
        "efri13",
        "efri14",
        "efri15",
        "efri18",
        "efri20",
        "efri26",
    ]

    for subject in tqdm(subjects):

        # subject identifiers
        session = "efri"
        task = "move"
        acquisition = "seeg"
        run = "01"
        kind = "ieeg"
        trial_id = 2

        bids_path = BIDSPath(
            subject=subject,
            session=session,
            task=task,
            acquisition=acquisition,
            run=run,
            suffix=kind,
            extension=".vhdr",
            root=bids_root,
        )

        # set time window
        # tmin, tmax = (-0.5, 1.0)
        tmin, tmax = (-0.75, 1.25)

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
        epochs = epochs.filter(l_freq=None, h_freq=fs / 2 - 1)
        # epochs = epochs.filter(l_freq=70, h_freq=200)

        # Downsample epochs to 500 Hz
        resample_rate = 500
        epochs = epochs.resample(resample_rate)

        if not os.path.exists(derivatives_path / subject):
            os.makedirs(derivatives_path / subject)

        fname = os.path.splitext(bids_path.basename)[0] + "-epo.fif"
        fpath = derivatives_path / subject / fname
        epochs.save(fpath, overwrite=True)

        print(f"{subject.upper()} epochs saved at {fpath}")
