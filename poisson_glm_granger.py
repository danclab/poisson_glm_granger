from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from joblib import Parallel, delayed
import numpy as np
from statsmodels.api import GLM, families

def cross_validate_window(source, target, data, lag, folds=5):
    """
    Perform cross-validation for a specific lag for a source-target pair using a Poisson GLM.

    Parameters:
    ----------
    source : int
        Index of the source neuron.
    target : int
        Index of the target neuron.
    data : ndarray
        3D array of shape (neurons, trials, time_steps) representing spike train data.
    lag : int
        The specific lag (time step) to use as a predictor.
    folds : int, optional
        The number of folds to use for K-fold cross-validation. Default is 5.

    Returns:
    -------
    float
        The average log-likelihood across all cross-validation folds for the source-target pair.
    """
    neurons, trials, time_steps = data.shape
    kf = KFold(n_splits=folds)

    avg_log_likelihood = 0

    for train_idx, test_idx in kf.split(range(trials)):
        train_data = data[:, train_idx, :]
        test_data = data[:, test_idx, :]

        target_train = train_data[target, :, lag:].reshape(-1)
        target_test = test_data[target, :, lag:].reshape(-1)

        # Extract predictors for the source neuron at the specific lag
        train_lagged = train_data[source, :, lag - lag:time_steps - lag].reshape(-1)
        test_lagged = test_data[source, :, lag - lag:time_steps - lag].reshape(-1)

        predictors_train = np.column_stack([np.ones_like(train_lagged), train_lagged])  # Add intercept
        predictors_test = np.column_stack([np.ones_like(test_lagged), test_lagged])    # Add intercept

        # Fit the Poisson GLM
        model = GLM(target_train, predictors_train, family=families.Poisson())
        results = model.fit()

        # Compute log-loss
        predicted_probs = results.predict(predictors_test)
        avg_log_likelihood += -log_loss(target_test, predicted_probs, labels=[0, 1])

    return avg_log_likelihood / folds



def compute_optimal_windows(data, window_range=(1, 10), folds=5, n_jobs=-1):
    """
    Compute the optimal history window size for each source-target pair.

    Parameters:
    ----------
    data : ndarray
        3D array of shape (neurons, trials, time_steps) representing spike train data.
    window_range : tuple of int, optional
        The range of history window sizes to evaluate (inclusive). Default is (1, 10).
    folds : int, optional
        The number of folds to use for K-fold cross-validation. Default is 5.
    n_jobs : int, optional
        The number of parallel jobs to run for cross-validation. Default is -1.

    Returns:
    -------
    dict
        A dictionary with keys `(source, target)` and values as a tuple:
        (optimal window size, cross-validated score).
    """
    neurons, _, _ = data.shape
    results = {}

    for target in range(neurons):
        for source in range(neurons):
            scores = Parallel(n_jobs=n_jobs)(
                delayed(cross_validate_window)(source, target, data, window, folds)
                for window in range(window_range[0], window_range[1] + 1)
            )

            best_cv_score = max(scores)
            best_window = window_range[0] + scores.index(best_cv_score)
            results[(source, target)] = (best_window, best_cv_score)

            print(f"Source {source} -> Target {target}: Optimal window = {best_window}, CV score = {best_cv_score:.4f}")

    return results


def compute_granger_causality(data, window_range=(1, 20), folds=10, n_jobs=-1, pairwise_windows=None,
                              gc_type='conditional'):
    """
    Compute Granger causality matrix using pair-specific optimal windows.

    Parameters:
    ----------
    data : ndarray
        3D array of shape (neurons, trials, time_steps) representing spike train data.
    window_range : tuple of int, optional
        The range of history window sizes to evaluate (inclusive). Default is (1, 10).
    folds : int, optional
        The number of folds to use for K-fold cross-validation to determine optimal history
        window sizes. Default is 5.
    n_jobs : int, optional
        The number of parallel jobs to run. Default is -1 (uses all available CPU cores).
    pairwise_windows: dict, optoinal
        A dictionary with keys `(source, target)` and values as the optimal window size
        for that pair, as returned by `compute_optimal_windows`. If None, this will be
        recomputed by calling compute_optimal_windows
    gc_type : str, optional
        Type of Granger causality to compute: "conditional" or "partial".
        Default is "conditional".

    Returns:
    -------
    dict
        A dictionary with keys `(source, target)` and values as the optimal window size
        for that pair, as returned by `compute_optimal_windows`.
    ndarray
        Granger causality matrix of shape (neurons, neurons), where each entry represents
        the causality score from one neuron to another.
    """
    neurons, trials, time_steps = data.shape

    if pairwise_windows is None:
        pairwise_windows = compute_optimal_windows(
            data,
            window_range=window_range,
            folds=folds,
            n_jobs=n_jobs
        )

    gc_matrix = np.zeros((neurons, neurons))
    signed_gc_matrix = np.zeros((neurons, neurons))

    # Parallel computation for each source-target pair
    gc_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_gc_for_pair)(data, source, target, pairwise_windows[(source, target)][0], gc_type=gc_type)
        for target in range(neurons)
        for source in range(neurons)
    )

    # Populate the GC matrix
    idx = 0
    for target in range(neurons):
        for source in range(neurons):
            gc_matrix[source, target] = gc_results[idx][0]
            signed_gc_matrix[source, target] = gc_results[idx][1]
            idx += 1

    return pairwise_windows, gc_matrix, signed_gc_matrix


def compute_gc_for_pair(data, source, target, lag, gc_type="conditional"):
    """
    Compute Granger causality score (conditional or partial) for a source-target pair.

    Parameters:
    ----------
    data : ndarray
        3D array of shape (neurons, trials, time_steps) representing spike train data.
    source : int
        Index of the source neuron.
    target : int
        Index of the target neuron.
    lag : int
        The specific lag (time step) to use for causality computation.
    gc_type : str, optional
        Type of Granger causality to compute: "conditional" or "partial".
        Default is "conditional".

    Returns:
    -------
    float
        The Granger causality score for the source-target pair.
    float
        The signed Granger causality score for the source-target pair.
    """
    neurons, trials, time_steps = data.shape
    target_spikes = data[target, :, lag:].reshape(-1)

    # Include all neurons except the source in the conditional model
    conditional_predictors = []
    for neuron in range(neurons):
        if neuron != source:
            lagged_data = data[neuron, :, lag - lag:time_steps - lag].reshape(-1)
            conditional_predictors.append(lagged_data)

    # Full model: source + other neurons
    lagged_source = data[source, :, lag - lag:time_steps - lag].reshape(-1)
    predictors_full = np.column_stack([np.ones_like(lagged_source), lagged_source] + conditional_predictors)

    # Reduced model: other neurons only
    predictors_reduced = np.column_stack([np.ones_like(lagged_source)] + conditional_predictors)

    # Fit full and reduced models
    full_model = GLM(target_spikes, predictors_full, family=families.Poisson()).fit()
    reduced_model = GLM(target_spikes, predictors_reduced, family=families.Poisson()).fit()

    # Conditional Granger causality
    if gc_type == "conditional":
        ll_full = full_model.llf
        ll_reduced = reduced_model.llf
        gc = 2 * (ll_full - ll_reduced)
    elif gc_type == "partial":
        # Conditional model: source + intercept only
        predictors_conditional = np.column_stack([np.ones_like(lagged_source), lagged_source])
        conditional_model = GLM(target_spikes, predictors_conditional, family=families.Poisson()).fit()

        # Compute likelihood ratios
        ll_full = full_model.llf
        ll_reduced = reduced_model.llf
        ll_conditional = conditional_model.llf

        # Compute Partial GC: full - (reduced - conditional)
        gc_full = 2 * (ll_full - ll_reduced)
        gc = gc_full - 2 * (ll_reduced - ll_conditional)
    else:
        raise ValueError(f"Invalid gc_type: {gc_type}. Must be 'conditional' or 'partial'.")

    # Determine interaction sign
    source_coeff = full_model.params[1]
    interaction_sign = np.sign(source_coeff)

    signed_gc = np.abs(gc) * interaction_sign
    return gc, signed_gc


def permutation_test(best_windows, gc_matrix, signed_gc_matrix, data, n_permutations=1000,
                     gc_type='conditional', n_jobs=-1):
    """
    Perform a one-tailed permutation test for signed Granger causality analysis.

    Parameters:
    ----------
    best_windows : dict
        Dictionary with keys `(source, target)` and values as the optimal lag for each pair.
    signed_gc_matrix : ndarray
        Signed Granger causality matrix computed on the original data.
    data : ndarray
        3D array of shape (neurons, trials, time_steps) representing spike train data.
    n_permutations : int, optional
        Number of permutations to perform for the test. Default is 1000.
    gc_type : str, optional
        Type of Granger causality to compute: "conditional" or "partial".
        Default is "conditional".
    n_jobs : int, optional
        Number of parallel jobs for the computation. Default is -1.

    Returns:
    -------
    ndarray
        Significance-thresholded signed Granger causality matrix.
    ndarray
        P-values for each connection.
    ndarray
        Null distributions for all connections.
    """
    neurons = data.shape[0]
    permuted_gc_matrices = np.zeros((n_permutations, neurons, neurons))

    def compute_permuted_gc(source, target, lag, perm_idx):
        """
        Compute permuted GC for a specific source-target pair by shuffling the source spikes.

        Parameters:
        ----------
        source : int
            Source neuron index.
        target : int
            Target neuron index.
        lag : int
            Optimal lag for this source-target pair.
        perm_idx : int
            Permutation index (unused, required for parallel execution).

        Returns:
        -------
        float
            Permuted signed GC value for the source-target pair.
        """
        # Shuffle source neuron spikes
        shuffled_data = data.copy()
        trials = shuffled_data.shape[1]
        for trial in range(trials):
            np.random.shuffle(shuffled_data[source, trial, :])  # Shuffle only the source neuron spikes

        # Recompute GC for the shuffled data
        return compute_gc_for_pair(shuffled_data, source, target, lag, gc_type=gc_type)[0]

    # Parallel computation of permuted GC matrices
    for target in range(neurons):
        for source in range(neurons):
            lag = best_windows[(source, target)][0]  # Optimal lag for this pair
            permuted_gc_values = Parallel(n_jobs=n_jobs)(
                delayed(compute_permuted_gc)(source, target, lag, perm)
                for perm in range(n_permutations)
            )
            permuted_gc_matrices[:, source, target] = permuted_gc_values

    # Compute p-values
    p_values = np.zeros_like(gc_matrix)
    for i in range(neurons):
        for j in range(neurons):
            observed_value = gc_matrix[i, j]
            null_values = permuted_gc_matrices[:, i, j]
            # One-tailed test for positive GC
            p_values[i, j] = (np.sum(null_values >= observed_value) + 1) / (n_permutations + 1)

    # Apply significance threshold (e.g., p < 0.05)
    alpha = 0.001
    significant_gc_matrix = np.where(p_values < alpha, signed_gc_matrix, 0)

    return significant_gc_matrix, p_values, permuted_gc_matrices
