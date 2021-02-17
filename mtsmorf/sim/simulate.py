import numpy as np


def _generate_correlated_samples(n_samples: int, mu=None, cov=None, random_state=None):
    """Generate multi-variate Gaussian correlated examples.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    mu : np.ndarray
        The mean vector.
    cov : np.ndarray
        The covariance matrix. Note it must be Positive Definite.
    random_state : int
        The random seed.

    Returns
    -------
    y : np.ndarray
        [3, n_samples] is the returned correlated example.
    """
    # set a random seed for the samples
    if random_state is not None:
        np.random.seed(random_state)

    # The desired mean values of the sample.
    if mu is None:
        mu = np.array([5.0,
                       0.0,
                       10.0])

    # The desired covariance matrix.
    if cov is None:
        # cov = np.array([
        #     [3, -2.75 * i, -2.00 * i],
        #     [-2.75 * i, 5, 1.50 * i],
        #     [-2.00 * i, 1.50 * i, 1]
        # ])
        cov = np.array([
            [3, 0, 0],
            [0, 5, 0],
            [0, 0, 1]
        ])
    # elif cov2 is None:
    #     r = np.array([
    #         [3, 0, 0],
    #         [0, 5, 0],
    #         [0, 0, 1]
    #     ])

    # Generate the random samples.
    y = np.random.multivariate_normal(mu, cov, size=n_samples).T

    return y


def generate_redundant_data(noise_dim: int = 3, n_samples: int = 100,
                            permutation_strategy: str = None,
                            mu=None, cov=None, indices=None,
                            random_state=None):
    """Generate iid random multivariate Gaussian data corrupted with redundant noise rows.

    Parameters
    ----------
    noise_dim :
    n_samples :
    permutation_strategy :
    mu :
    cov :
    indices :
    random_state :

    Returns
    -------

    """
    # generate noise vector
    y_noise = np.random.random((noise_dim, n_samples))

    # generate the data
    y = _generate_correlated_samples(n_samples=n_samples, mu=mu, cov=cov, random_state=random_state)
    data_dim = y.shape[0]

    data = np.zeros((data_dim + noise_dim, n_samples))
    if permutation_strategy is None:
        data = np.vstack((y, y_noise))
    elif permutation_strategy == 'alternate':
        if y_noise.size != y.size:
            raise RuntimeError(f'For permutation strategy "alternate", '
                               f'need to have noise and data dimension be '
                               f'equal.')
        mult_factor = 2
        # mult_factor = int(y_noise.shape[0] / y.shape[0])
        # print(mult_factor)
        data[1::mult_factor, :] = y
        data[::mult_factor, :] = y_noise
        # data = np.ravel([y, y_noise], order="F").reshape(n_samples, data_dim + noise_dim).T
    elif permutation_strategy == 'random':
        signal_inds = indices[:data_dim]
        other_inds = indices[data_dim:]
        data[signal_inds, :] = y
        data[other_inds, :] = y_noise
    return data


