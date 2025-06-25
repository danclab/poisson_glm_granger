from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from joblib import Parallel, delayed
import numpy as np
from statsmodels.api import GLM, families
from statsmodels.stats.multitest import multipletests

def cross_validate_lag(source, target, data, lag, folds=10, family=None, family_kwargs=None):
    neurons, trials, time_steps = data.shape
    kf = KFold(n_splits=folds)
    avg_log_likelihood = 0

    if family is None:
        family = families.NegativeBinomial
    if family_kwargs is None:
        family_kwargs = {"alpha": 1.0}

    for train_idx, test_idx in kf.split(range(trials)):
        train_data = data[:, train_idx, :]
        test_data = data[:, test_idx, :]

        target_train = train_data[target, :, lag:].reshape(-1)
        target_test = test_data[target, :, lag:].reshape(-1)

        train_lagged = train_data[source, :, lag - lag:time_steps - lag].reshape(-1)
        test_lagged = test_data[source, :, lag - lag:time_steps - lag].reshape(-1)

        predictors_train = np.column_stack([np.ones_like(train_lagged), train_lagged])
        predictors_test = np.column_stack([np.ones_like(test_lagged), test_lagged])

        model = GLM(target_train, predictors_train, family=family(**family_kwargs))
        results = model.fit(method="lbfgs")

        predicted_probs = results.predict(predictors_test)
        avg_log_likelihood += -log_loss(target_test, predicted_probs, labels=[0, 1])

    return avg_log_likelihood / folds

def compute_optimal_lags(data, lags=None, folds=10, n_jobs=-1, family=None, family_kwargs=None):
    if lags is None:
        lags = []
    neurons, _, _ = data.shape

    def compute_score_for_lag(source, target):
        scores = [
            cross_validate_lag(source, target, data, lag, folds, family, family_kwargs)
            for lag in lags
        ]
        best_idx = np.argmax(scores)
        best_cv_score = scores[best_idx]
        best_lag = lags[best_idx]
        return (source, target, best_lag, best_cv_score)

    source_target_pairs = [(source, target) for source in range(neurons) for target in range(neurons)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_score_for_lag)(source, target)
        for source, target in source_target_pairs
    )

    results_dict = {(source, target): (best_lag, best_cv_score) for source, target, best_lag, best_cv_score in results}
    return results_dict

def compute_gc_for_pair(data, source, target, lag, family=None, family_kwargs=None):
    neurons, trials, time_steps = data.shape
    target_spikes = data[target, :, lag:].reshape(-1)

    if family is None:
        family = families.NegativeBinomial
    if family_kwargs is None:
        family_kwargs = {"alpha": 1.0}

    conditional_predictors = []
    for neuron in range(neurons):
        if neuron != source:
            lagged_data = data[neuron, :, lag - lag:time_steps - lag].reshape(-1)
            conditional_predictors.append(lagged_data)

    lagged_source = data[source, :, lag - lag:time_steps - lag].reshape(-1)

    gc = 0
    signed_gc = 0
    if not np.all(lagged_source == 0) or np.all(target_spikes == 0):
        conditional_predictors = np.column_stack(conditional_predictors)
        predictors_full = np.column_stack([np.ones_like(lagged_source), lagged_source, conditional_predictors])
        predictors_reduced = np.column_stack([np.ones_like(lagged_source), conditional_predictors])

        predictors_full = predictors_full[:, ~np.all(predictors_full == 0, axis=0)]
        predictors_reduced = predictors_reduced[:, ~np.all(predictors_reduced == 0, axis=0)]

        try:
            full_model = GLM(target_spikes, predictors_full, family=family(**family_kwargs)).fit(method="lbfgs")
            reduced_model = GLM(target_spikes, predictors_reduced, family=family(**family_kwargs)).fit(method="lbfgs")
            ll_full = full_model.llf
            ll_reduced = reduced_model.llf
            gc = 2 * (ll_full - ll_reduced)
            source_coeff = full_model.params[1]
            signed_gc = np.abs(gc) * np.sign(source_coeff)
        except Exception as e:
            print(f"Error fitting from {source} to {target} at lag {lag}: {str(e)}")
            raise ValueError("stop")

    return gc, signed_gc

def compute_granger_causality(data, lags=None, folds=10, n_jobs=-1, pairwise_lags=None, family=None, family_kwargs=None):
    if lags is None:
        lags = []
    neurons, trials, time_steps = data.shape
    print(f'Data contains {neurons} neurons, {trials} trials, and {time_steps} time steps.')

    if pairwise_lags is None:
        pairwise_lags = compute_optimal_lags(data, lags=lags, folds=folds, n_jobs=n_jobs, family=family, family_kwargs=family_kwargs)

    gc_matrix = np.zeros((neurons, neurons))
    signed_gc_matrix = np.zeros((neurons, neurons))

    gc_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_gc_for_pair)(data, source, target, pairwise_lags[(source, target)][0], family, family_kwargs)
        for target in range(neurons)
        for source in range(neurons)
    )

    idx = 0
    for target in range(neurons):
        for source in range(neurons):
            gc_matrix[source, target] = gc_results[idx][0]
            signed_gc_matrix[source, target] = gc_results[idx][1]
            idx += 1

    return pairwise_lags, gc_matrix, signed_gc_matrix

def filter_indirect_connections(pairwise_lags, gc_matrix, signed_gc_matrix, dominance_threshold=0.5, uniqueness_threshold=0.1):
    neurons = gc_matrix.shape[0]
    filtered_gc_matrix = gc_matrix.copy()
    filtered_signed_gc_matrix = signed_gc_matrix.copy()

    for target in range(neurons):
        for source in range(neurons):
            if source == target:
                continue

            direct_lag = pairwise_lags[(source, target)][0]
            direct_gc = gc_matrix[source, target]
            direct_signed_gc = signed_gc_matrix[source, target]

            total_indirect_gc = 0
            total_indirect_signed_gc = 0
            for intermediate in range(neurons):
                if intermediate == source or intermediate == target:
                    continue

                source_to_intermediate_lag = pairwise_lags[(source, intermediate)][0]
                intermediate_to_target_lag = pairwise_lags[(intermediate, target)][0]

                if source_to_intermediate_lag + intermediate_to_target_lag == direct_lag:
                    indirect_gc = gc_matrix[source, intermediate] * gc_matrix[intermediate, target]
                    indirect_signed_gc = signed_gc_matrix[source, intermediate] * signed_gc_matrix[intermediate, target]
                    total_indirect_gc += indirect_gc
                    total_indirect_signed_gc += indirect_signed_gc

            indirect_fraction = total_indirect_gc / (direct_gc + 1e-10)

            if indirect_fraction >= dominance_threshold:
                remaining_gc = direct_gc - total_indirect_gc
                if remaining_gc < uniqueness_threshold * direct_gc:
                    filtered_gc_matrix[source, target] = 0
                    filtered_signed_gc_matrix[source, target] = 0
                else:
                    filtered_gc_matrix[source, target] = remaining_gc
                    filtered_signed_gc_matrix[source, target] = direct_signed_gc - total_indirect_signed_gc

    return filtered_gc_matrix, filtered_signed_gc_matrix

def permutation_test(pairwise_lags, gc_matrix, signed_gc_matrix, data, n_permutations=1000, n_jobs=-1, alpha=0.05, family=None, family_kwargs=None):
    neurons = data.shape[0]
    permuted_gc_matrices = np.zeros((n_permutations, neurons, neurons))

    def compute_permuted_gc(source, target, lag):
        shuffled_data = data.copy()
        trials = shuffled_data.shape[1]
        if source == target:
            for trial in range(trials):
                np.random.shuffle(shuffled_data[source, trial, :])
        else:
            permuted_trials = np.random.permutation(trials)
            shuffled_data[source, :, :] = shuffled_data[source, permuted_trials, :]

        return compute_gc_for_pair(shuffled_data, source, target, lag, family, family_kwargs)[0]

    for target in range(neurons):
        for source in range(neurons):
            lag = pairwise_lags[(source, target)][0]
            permuted_gc_values = Parallel(n_jobs=n_jobs)(
                delayed(compute_permuted_gc)(source, target, lag)
                for _ in range(n_permutations)
            )
            permuted_gc_matrices[:, source, target] = permuted_gc_values

    p_values = np.zeros_like(gc_matrix)
    for i in range(neurons):
        for j in range(neurons):
            observed_value = gc_matrix[i, j]
            null_values = permuted_gc_matrices[:, i, j]
            p_values[i, j] = (np.sum(null_values >= observed_value) + 1) / (n_permutations + 1)

    flattened_p_values = p_values.flatten()
    _, corrected_p_values, _, _ = multipletests(flattened_p_values, alpha=alpha, method='fdr_bh')
    corrected_p_values = corrected_p_values.reshape(p_values.shape)
    significant_gc_matrix = np.where(corrected_p_values < alpha, signed_gc_matrix, 0)

    return significant_gc_matrix, corrected_p_values, permuted_gc_matrices
