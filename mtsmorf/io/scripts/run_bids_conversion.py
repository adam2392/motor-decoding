"""API for converting files to BIDS format."""
import logging
import os
import tempfile
from pathlib import Path
from pprint import pprint
from typing import Union

import mne
from mne_bids import BIDSPath, write_raw_bids
from mne_bids.sidecar_updates import _write_json
from natsort import natsorted
from tqdm import tqdm
import scipy
from mtsmorf.io.bids_conversion import (
    _convert_mat_to_raw,
    _convert_trial_info_war,
    _create_electrodes_tsv,
    _convert_trial_info_move,
    _append_anat_to_channels,
    _get_setup_fname, _get_misc_fname, _get_xy_fname,
    read_matlab, MatReader
)

# from mtsmorf.io.utils import append_original_fname_to_scans
from mtsmorf.io.utils import append_original_fname_to_scans

logger = logging.getLogger(__name__)


def convert_behav_to_bids(bids_path, source_fpath):
    setup_src_fpath = _get_setup_fname(source_fpath)
    xy_src_fpath = _get_xy_fname(source_fpath)

    # create behavioral tsv files from the data
    if task == "war":
        bids_basename = bids_path.basename
        _convert_trial_info_war(source_fpath, bids_basename, bids_root)
    elif task == "move":
        _convert_trial_info_move(source_fpath, bids_path)

    # read in the XY data
    xy_dict = read_matlab(xy_src_fpath)

    # create raw array
    ch_names = ['x', 'y']
    xy_data = xy_dict["XYdata"]
    sfreq = xy_dict["Fs"]
    if xy_data.shape[1] == len(ch_names):
        xy_data = xy_data.T

    # get the raw Array
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="misc")
    raw = mne.io.RawArray(xy_data, info=info)

    # set a measurement date to allow anonymization to run
    # raw.set_meas_date(meas_date=datetime.datetime.now(tz=datetime.timezone.utc))
    # mne.io.anonymize_info(raw.info)
    raw.info["line_freq"] = None

    # save the raw file
    behav_xy_fpath = bids_path.copy().update(
        recording='xy',
        suffix='physio',
        extension='.tsv.gz',
        check=False
    )
    # save as tsv
    behav_df = raw.to_data_frame(index='time', scalings={})

    behav_sidecar = behav_xy_fpath.copy().update(
        extension='.json'
    )
    behav_json = {
        'SamplingFrequency': raw.info['sfreq'],
        'StartTime': 0,
        'Columns': ['x', 'y'],
        'x': {
            'Units': 'mm',
        },
        'y': {
            'Units': 'mm',
        }
    }
    behav_df.to_csv(behav_xy_fpath,
                    index=None,
                    compression='gzip',
                    sep='\t')

    _write_json(behav_sidecar, behav_json, overwrite=True)

def convert_mat_to_bids(
    bids_path,
    source_fpath: Union[str, os.PathLike],
    overwrite=True,
    verbose=False,
) -> str:
    """Run Bids conversion pipeline given filepaths."""
    print("Converting ", source_fpath, "to ", bids_path)

    # handle conversion to raw Array
    raw = _convert_mat_to_raw(source_fpath)
    raw = None
    # add trial info and save it
    if task == "war":
        raw = _convert_trial_info_war(source_fpath, bids_path, bids_root, raw)
    elif task == "move":
        raw = _convert_trial_info_move(source_fpath, bids_path=bids_path, raw=raw)

    # create electrodes tsv file with anatomy
    # _create_electrodes_tsv(source_fpath, bids_basename, bids_root)

    # get events
    # events, events_id = mne.events_from_annotations(raw)
    #
    # with tempfile.TemporaryDirectory() as tmproot:
    #     tmpfpath = os.path.join(tmproot, "tmp_raw.fif")
    #     raw.save(tmpfpath)
    #     raw = mne.io.read_raw_fif(tmpfpath)
    #
    #     # Get acceptable range of days back and pick random one
    #     daysback = 0
    #
    #     # write to BIDS
    #     bids_root = write_raw_bids(
    #         raw,
    #         bids_basename,
    #         bids_root=str(bids_root),
    #         overwrite=overwrite,
    #         anonymize=dict(daysback=daysback, keep_his=False),
    #         events_data=events,
    #         event_id=events_id,
    #         verbose=verbose,
    #     )

    # append data to channels.tsv
    # _append_anat_to_channels(source_fpath, bids_path)

    return bids_root


def _main(
    bids_root, source_path, subject_ids, acquisition, task, session
):  # pragma: no cover
    """Run Bids Conversion script to be updated.

    Just to show example run locally.
    """
    ext = "mat"

    # set BIDS kind based on acquistion
    if acquisition in ["ecog", "seeg", "ieeg"]:
        datatype = "ieeg"
    elif acquisition in ["eeg"]:
        datatype = "eeg"

    # go through each subject
    for subject in subject_ids:
        # get specific files
        subj_dir = Path(source_path / subject)
        rawfiles = [
            x
            for x in subj_dir.glob(f"*.{ext}")
            if task in x.name.lower()
            if "raw" in x.name.lower()
            if not x.name.startswith(".")  # make sure not a cached hidden file
        ]

        # make subject an efri number
        subject = subject.replace("SUBJECT", "efri")
        pprint(f"In {subj_dir} found {rawfiles}")

        if rawfiles == []:
            continue

        # run BIDs conversion for each separate dataset
        for run_id, fpath in enumerate(tqdm(natsorted(rawfiles)), start=1):
            logger.info(f"Running run id: {run_id}, with filepath: {fpath}")
            bids_path = BIDSPath(
                subject=subject, session=session, task=task, acquisition=acquisition, run=run_id,
                suffix=datatype, datatype=datatype, extension='.vhdr',
                root=bids_root
            )
            # if any(bids_basename in x.name for x in subj_dir.rglob("*.vhdr")):
            #     continue

            # convert mat raw data into BIDs
            # convert_mat_to_bids(bids_path, fpath, overwrite=True)

            # convert and store the bhariovial tsv
            convert_behav_to_bids(bids_path, source_fpath=fpath)

            # append scans original filenames
            # append_original_fname_to_scans(
            #     os.path.basename(fpath), bids_root, bids_fname
            # )

        # break


if __name__ == "__main__":
    # bids root to write BIDS data to
    # bids_root = Path("/Users/adam2392/Dropbox/efri/")
    # bids_root = Path('/Users/adam2392/OneDrive - Johns Hopkins/efri/')
    bids_root = Path("/home/adam2392/hdd/Dropbox/efri/")

    # path to excel layout file - would be changed to the datasheet locally
    # define BIDS identifiers
    acquisition = "seeg"
    task = "move"
    session = "efri"

    # path to original source data
    source_path = Path(bids_root / "sourcedata")

    # HACK: get all subject ids within sourcedata
    subject_ids = natsorted(
        [
            x.name  # .replace("SUBJECT", '')
            for x in source_path.iterdir()
            if not x.as_posix().startswith(".")
            if x.is_dir()
        ]
    )
    # subject_ids = [
    #     'efri07'
    # ]
    print(subject_ids)

    # run main bids conversion
    _main(
        bids_root, source_path, subject_ids, acquisition, task, session,
    )
