from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from joblib import Parallel, delayed
import numpy as np
from statsmodels.api import GLM, families
from statsmodels.stats.multitest import multipletests

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


def compute_granger_causality(data, window_range=(1, 20), folds=10, n_jobs=-1, pairwise_windows=None):
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
        delayed(compute_gc_for_pair)(data, source, target, pairwise_windows[(source, target)][0])
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


def filter_indirect_connections(data, best_windows, gc_matrix, signed_gc_matrix, dominance_threshold=0.5, uniqueness_threshold=0.1):
    """
    Filter out indirect influences from the Granger causality matrix.

    Parameters:
    ----------
    data : ndarray
        3D array of shape (neurons, trials, time_steps) representing spike train data.
    best_windows : dict
        Optimal window sizes for each pair as computed by `compute_optimal_windows`.
    gc_matrix : ndarray
        The original Granger causality matrix.
    signed_gc_matrix : ndarray
        The signed Granger causality matrix.
    dominance_threshold : float, optional
        Fraction of GC that must be explained by indirect paths to warrant filtering. Default is 0.5.
    uniqueness_threshold : float, optional
        Minimum fraction of direct GC that must remain unique after accounting for indirect contributions. Default is 0.1.

    Returns:
    -------
    filtered_gc_matrix : ndarray
        GC matrix with indirect influences adjusted.
    filtered_signed_gc_matrix : ndarray
        Signed GC matrix with indirect influences adjusted.
    """
    neurons = data.shape[0]
    filtered_gc_matrix = gc_matrix.copy()
    filtered_signed_gc_matrix = signed_gc_matrix.copy()

    for target in range(neurons):
        for source in range(neurons):
            if source == target:
                continue

            # Direct GC for the source-target pair
            direct_window = best_windows[(source, target)][0]
            direct_gc = gc_matrix[source, target]
            direct_signed_gc = signed_gc_matrix[source, target]

            # Accumulate indirect contributions
            total_indirect_gc = 0
            total_indirect_signed_gc = 0
            for intermediate in range(neurons):
                if intermediate == source or intermediate == target:
                    continue

                # Intermediate path windows
                source_to_intermediate_window = best_windows[(source, intermediate)][0]
                intermediate_to_target_window = best_windows[(intermediate, target)][0]

                # Validate temporal alignment for indirect paths
                if source_to_intermediate_window + intermediate_to_target_window == direct_window:
                    indirect_gc = gc_matrix[source, intermediate] * gc_matrix[intermediate, target]
                    indirect_signed_gc = (
                        signed_gc_matrix[source, intermediate] * signed_gc_matrix[intermediate, target]
                    )
                    total_indirect_gc += indirect_gc
                    total_indirect_signed_gc += indirect_signed_gc

            # Calculate the fraction of direct GC explained by indirect paths
            indirect_fraction = total_indirect_gc / (direct_gc + 1e-10)  # Prevent division by zero

            # Apply filtering logic
            if indirect_fraction >= dominance_threshold:
                remaining_gc = direct_gc - total_indirect_gc
                if remaining_gc < uniqueness_threshold * direct_gc:
                    # Fully filter if little unique GC remains
                    filtered_gc_matrix[source, target] = 0
                    filtered_signed_gc_matrix[source, target] = 0
                else:
                    # Otherwise, adjust the direct GC to reflect unique contribution
                    filtered_gc_matrix[source, target] = remaining_gc
                    filtered_signed_gc_matrix[source, target] = direct_signed_gc - total_indirect_signed_gc

    return filtered_gc_matrix, filtered_signed_gc_matrix





def compute_gc_for_pair(data, source, target, lag):
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
    ll_full = full_model.llf
    ll_reduced = reduced_model.llf
    gc = 2 * (ll_full - ll_reduced)

    # Determine interaction sign
    source_coeff = full_model.params[1]
    interaction_sign = np.sign(source_coeff)

    signed_gc = np.abs(gc) * interaction_sign
    return gc, signed_gc


def permutation_test(best_windows, gc_matrix, signed_gc_matrix, data, n_permutations=1000,
                     n_jobs=-1, alpha=0.05):
    """
    Perform a one-tailed permutation test for signed Granger causality analysis with FDR correction.

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
    n_jobs : int, optional
        Number of parallel jobs for the computation. Default is -1.
    alpha : float, optional
        Significance level for FDR correction. Default is 0.05.

    Returns:
    -------
    ndarray
        FDR-corrected significant signed Granger causality matrix.
    ndarray
        Adjusted p-values for each connection after FDR correction.
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
            Optimal lag for this pair.
        perm_idx : int
            Permutation index (unused, required for parallel execution).

        Returns:
        -------
        float
            Permuted signed GC value for the source-target pair.
        """
        shuffled_data = data.copy()
        trials = shuffled_data.shape[1]
        if source == target:
            for trial in range(trials):
                np.random.shuffle(shuffled_data[source, trial, :])
        else:
            permuted_trials = np.random.permutation(trials)
            shuffled_data[source, :, :] = shuffled_data[source, permuted_trials, :]

        return compute_gc_for_pair(shuffled_data, source, target, lag)[0]

    # Parallel computation of permuted GC matrices
    for target in range(neurons):
        for source in range(neurons):
            lag = best_windows[(source, target)][0]
            permuted_gc_values = Parallel(n_jobs=n_jobs)(
                delayed(compute_permuted_gc)(source, target, lag, perm)
                for perm in range(n_permutations)
            )
            permuted_gc_matrices[:, source, target] = permuted_gc_values

    # Compute raw p-values
    p_values = np.zeros_like(gc_matrix)
    for i in range(neurons):
        for j in range(neurons):
            observed_value = gc_matrix[i, j]
            null_values = permuted_gc_matrices[:, i, j]
            p_values[i, j] = (np.sum(null_values >= observed_value) + 1) / (n_permutations + 1)

    # FDR correction
    flattened_p_values = p_values.flatten()
    _, corrected_p_values, _, _ = multipletests(flattened_p_values, alpha=alpha, method='fdr_bh')
    corrected_p_values = corrected_p_values.reshape(p_values.shape)

    # Apply FDR threshold to determine significance
    significant_gc_matrix = np.where(corrected_p_values < alpha, signed_gc_matrix, 0)

    return significant_gc_matrix, corrected_p_values, permuted_gc_matrices

