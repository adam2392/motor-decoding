from rerf.rerfClassifier import rerfClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def preprocess_epochs(epochs, resample_rate=500):
    """Preprocess mne.Epochs object in the following way:
    1. Low-pass filter up to Nyquist frequency
    2. Downsample data to 500 Hz 
    """
    # Low-pass filter up to sfreq/2
    fs = epochs.info["sfreq"]
    new_epochs = epochs.filter(l_freq=1, h_freq=fs / 2 - 1)

    # Downsample epochs to 500 Hz
    new_epochs = new_epochs.resample(resample_rate)

    return new_epochs


def initialize_classifiers(image_height, image_width, n_jobs=1, random_state=None):
    """Initialize a list of classifiers to be compared."""

    mtsmorf = rerfClassifier(
        projection_matrix="MT-MORF",
        max_features="auto",
        n_jobs=n_jobs,
        random_state=random_state,
        image_height=image_height,
        image_width=image_width,
    )

    srerf = rerfClassifier(
        projection_matrix="S-RerF",
        max_features="auto",
        n_jobs=n_jobs,
        random_state=random_state,
        image_height=image_height,
        image_width=image_width,
    )

    lr = LogisticRegression(random_state=random_state)
    rf = RandomForestClassifier(random_state=random_state)
    dummy = DummyClassifier(strategy="most_frequent", random_state=random_state)

    clfs = [mtsmorf, srerf, lr, rf, dummy]

    return clfs