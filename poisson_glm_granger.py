from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from joblib import Parallel, delayed
import numpy as np
from statsmodels.api import GLM, families


def cross_validate_window(data, window, folds=5):
    """
    Perform cross-validation for a specific history window size using a Poisson GLM.

    This function evaluates the average cross-validated log-likelihood for a given
    window size by splitting the data into training and testing sets. It fits a
    Poisson GLM for each target neuron using lagged predictors from the data.

    Parameters:
    ----------
    data : ndarray
        3D array of shape (neurons, trials, time_steps) representing spike train data
        for multiple neurons and trials.
    window : int
        The number of past time steps (history window) to use as predictors in the
        Granger causality analysis.
    folds : int, optional
        The number of folds to use for K-fold cross-validation. Default is 5.

    Returns:
    -------
    float
        The average log-likelihood across all cross-validation folds, averaged over
        all neurons and trials.
    """
    neurons, trials, time_steps = data.shape
    kf = KFold(n_splits=folds)

    # Average log likelihood over all folds
    avg_log_likelihood = 0

    # For each split
    for train_idx, test_idx in kf.split(range(trials)):
        # Get train and test data
        train_data = data[:, train_idx, :]
        test_data = data[:, test_idx, :]

        # For each target neuron
        for target in range(neurons):
            # Get the train and test spikes for the target
            target_train = train_data[target, :, window:].reshape(-1)
            target_test = test_data[target, :, window:].reshape(-1)

            # Design matrix for the train and test set
            predictors_train = []
            predictors_test = []

            # For each source neuron
            for i in range(neurons):
                # Iterate over lags
                for lag in range(1, window + 1):
                    # Get vectors of source neuron spiking for train and test set
                    # over all time steps and trials
                    train_lagged = train_data[i, :, window - lag:time_steps - lag].reshape(-1)
                    test_lagged = test_data[i, :, window - lag:time_steps - lag].reshape(-1)
                    predictors_train.append(train_lagged)
                    predictors_test.append(test_lagged)

            # Transpose so it is total time steps x (neuron * lags)
            predictors_train = np.array(predictors_train).T
            predictors_test = np.array(predictors_test).T

            # Add intercepts
            intercept_train = np.ones((predictors_train.shape[0], 1))
            predictors_train = np.hstack((intercept_train, predictors_train))
            intercept_test = np.ones((predictors_test.shape[0], 1))
            predictors_test = np.hstack((intercept_test, predictors_test))

            # Fit model on training set
            model = GLM(target_train, predictors_train, family=families.Poisson())
            results = model.fit()

            # Compute log likelihood on test set
            predicted_probs = results.predict(predictors_test)
            avg_log_likelihood += -log_loss(target_test, predicted_probs, labels=[0, 1])

    # Return average log likelihood over folds
    return avg_log_likelihood / folds


def poisson_glm_granger(data, window_range=(1, 10), folds=5, n_jobs=-1):
    """
    Perform Granger causality analysis using a Poisson GLM with cross-validation
    to select the optimal history window size.

    This function computes the Granger causality matrix for a dataset by:
    1. Performing cross-validation across a range of history window sizes to find the
       window size that maximizes the average log-likelihood.
    2. Using the optimal window size to compute the Granger causality matrix for the data.

    Parameters:
    ----------
    data : ndarray
        3D array of shape (neurons, trials, time_steps) representing spike train data
        for multiple neurons and trials.
    window_range : tuple of int, optional
        The range of history window sizes to evaluate (inclusive). Default is (1, 10).
    folds : int, optional
        The number of folds to use for K-fold cross-validation. Default is 5.
    n_jobs : int, optional
        The number of parallel jobs to run for cross-validation. Default is -1
        (uses all available CPU cores).

    Returns:
    -------
    tuple:
        - gc_matrix : ndarray
            Granger causality matrix of shape (neurons, neurons), where each entry
            represents the causality score from one neuron to another.
        - best_window : int
            The optimal history window size selected via cross-validation.
        - best_cv_score : float
            The cross-validated log-likelihood score for the optimal history window size.
    """
    neurons, trials, time_steps = data.shape
    best_window = None
    best_cv_score = -np.inf

    # Parallelize cross-validation for each window size using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(cross_validate_window)(data, window, folds) for window in range(window_range[0], window_range[1] + 1)
    )

    # Find the best window size based on CV score
    for window, cv_score in zip(range(window_range[0], window_range[1] + 1), results):
        if cv_score > best_cv_score:
            best_cv_score = cv_score
            best_window = window

    # After selecting the best window, compute the Granger causality matrix
    gc_matrix = np.zeros((neurons, neurons))
    # For each target neuron
    for target in range(neurons):
        # Get target neuron spikes
        target_spikes = data[target, :, best_window:].reshape(-1)

        # Create design matrix
        predictors_full = []
        # For each source neuron
        for i in range(neurons):
            # Iterate over lags
            for lag in range(1, best_window + 1):
                # Get vector of source neuron spiking over all time steps and trials
                lagged_data = data[i, :, best_window - lag:time_steps - lag].reshape(-1)
                predictors_full.append(lagged_data)

        # Transpose so it is total time steps x (neuron * lags)
        predictors_full = np.array(predictors_full).T

        # Add intercept
        intercept = np.ones((predictors_full.shape[0], 1))
        predictors_full_with_intercept = np.hstack((intercept, predictors_full))

        # Fit model
        full_model = GLM(target_spikes, predictors_full_with_intercept, family=families.Poisson()).fit()

        # For each source neuron
        for source in range(neurons):
            # Remove source neuron spiking from design matrix
            predictors_reduced = np.delete(predictors_full, slice(source * best_window, (source + 1) * best_window),
                                           axis=1)
            # Add intercept
            predictors_reduced_with_intercept = np.hstack((intercept, predictors_reduced))
            # Fit reduced model without this neuron's data
            reduced_model = GLM(target_spikes, predictors_reduced_with_intercept, family=families.Poisson()).fit()

            # Compute the likelihood ratio
            ll_full = full_model.llf
            ll_reduced = reduced_model.llf
            llr = 2 * (ll_full - ll_reduced)

            # Figure out the sign of the interaction from the fitted model coefficients
            source_coeff_start = 1 + source * best_window
            source_coeff_end = source_coeff_start + best_window
            source_coeff = full_model.params[source_coeff_start:source_coeff_end]
            interaction_sign = np.sign(np.sum(source_coeff))

            # Update connectivity matrix
            gc_matrix[source, target] = llr * interaction_sign

    return gc_matrix, best_window, best_cv_score