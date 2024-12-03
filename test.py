import numpy as np
import matplotlib.pyplot as plt

from poisson_glm_granger import poisson_glm_granger


# Function to generate synthetic spike train data for trials
def generate_synthetic_spike_data_trials(neurons, trials, time_steps, causal_matrix, base_rate=0.01, noise_level=0.1):
    data = np.zeros((neurons, trials, time_steps))
    for trial in range(trials):
        for t in range(1, time_steps):
            for i in range(neurons):
                firing_rate = base_rate
                for j in range(neurons):
                    if causal_matrix[j, i] != 0:
                        firing_rate += causal_matrix[j, i] * data[j, trial, t - 1]
                firing_prob = max(0, min(1, firing_rate + np.random.normal(0, noise_level)))
                data[i, trial, t] = 1 if np.random.rand() < firing_prob else 0
    return data

# Simulate synthetic spike train data for trials
causal_matrix = np.array([
    [0.0, 0.5, 0.0],
    [0.0, 0.0, -0.5],
    [0.0, 0.0, 0.0]
])
simulated_data_trials = generate_synthetic_spike_data_trials(
    neurons=3, trials=100, time_steps=100, causal_matrix=causal_matrix, base_rate=0.01, noise_level=0.1
)

# Analyze Granger causality with cross-validation
best_gc_matrix, best_window, best_cv_score = poisson_glm_granger(
    simulated_data_trials,
    window_range=(1, 10),
    folds=5
)

# Display results
plt.imshow(best_gc_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Signed Granger Causality')
plt.title(f"Best Granger Causality Matrix (Window={best_window}, CV Score={best_cv_score:.2f})")
plt.xlabel("Target Neuron")
plt.ylabel("Source Neuron")
plt.show()

# Print best Granger causality matrix
print(f"Best Granger Causality Matrix (Window={best_window}, CV Score={best_cv_score:.2f}):")
print(best_gc_matrix)