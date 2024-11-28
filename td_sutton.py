from math import sqrt

import matplotlib
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.interpolate import make_interp_spline, BSpline

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def execute_sequence(steps, weights, lamda, alpha):
    delta_w = [0.0] * 5
    z = 1 if (steps[-1][-1] == 1) else 0
    training_sample = steps[:, 1:6]
    seq_length = len(steps) - 1

    for t in range(seq_length):
        if t == seq_length - 1:
            delta_p = z - np.dot(weights, training_sample[t])
        else:
            delta_p = np.dot(weights, training_sample[t + 1]) - np.dot(weights, training_sample[t])

        for k in range(1, t + 1):
            delta_w_pk = np.array([i * (lamda ** (t - k)) for i in training_sample[k]], dtype='float64')
            delta_w = np.add(delta_w, (alpha * delta_p) * delta_w_pk)

    return delta_w


def generate_training_episodes(n=100):
    training_episodes = []
    for i in range(n):
        # Every walk begins in the center state D.
        vector_xi = np.array([0, 0, 0, 1, 0, 0, 0])
        current_state = 3
        # If either edge state (A or G) is entered, the walk terminates
        while current_state != 0 and current_state != 6:
            # Either to the right or to the left with equal probability uniformly
            if np.random.uniform() < 0.5:
                current_state = current_state - 1
            else:
                current_state = current_state + 1
            new_state = np.array([0, 0, 0, 0, 0, 0, 0])
            new_state[current_state] = 1
            # Adding this next step to the sequence
            vector_xi = np.vstack([vector_xi, new_state])

        training_episodes.append(vector_xi)

    return training_episodes


def run_example62_values(training_episodes, alpha):
    # True predications/weights for each of the states are given as  [1/6, 1/3, 1/2, 2/3, 5/6] for B,C,D,E,F
    ideal_predictions = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 5.0 / 6.0]
    errors = []
    lamda = 0.0  # TD(0) used in example 6.2
    value_episodes = [0, 1, 10, 100, 500, 1000]
    pred_values = dict()
    pred_values[0] = [0.5] * 5
    curr_weights = [0.5] * 5
    for i, sample_steps in enumerate(training_episodes):
        delta_w = execute_sequence(sample_steps, curr_weights, lamda, alpha)
        curr_weights = np.add(curr_weights, delta_w)

        errors.append(sqrt(mean_squared_error(curr_weights, ideal_predictions)))
        if i + 1 in value_episodes:
            pred_values[i + 1] = curr_weights
            # print("'{}':{}".format(i + 1, curr_weights))
    return pred_values, errors


def plot_value_graph(pred_values):
    sns.set(style="ticks")
    plt.figure(figsize=(7, 6))
    value_episodes = [1, 10, 100, 1000]
    ideal_predictions = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 5.0 / 6.0]
    plt.plot(ideal_predictions, label='True value', marker='x', color='blueviolet', linestyle='--', linewidth=0.9)
    colors = ['lightgrey', 'olivedrab', 'lightsteelblue', 'mediumvioletred']
    markers = ['o', 'x', '+', '>']
    linestyles = ['-.', ':', '--', '-']
    for i, ep_length in enumerate(value_episodes):
        plt.plot(pred_values[ep_length], label="episodes={}".format(str(ep_length)), marker=markers[i], color=colors[i], linestyle=linestyles[i], linewidth=0.9)
    plt.xlabel('State')
    plt.ylabel('Estimated Value')
    plt.xticks(np.arange(5), ['B', 'C', 'D', 'E', 'F'])
    plt.title('Convergence towards true value with episodic TD(0) Learning')
    plt.legend(loc="upper left")
    plt.savefig("output/random_walk_values_ex62.png")
    plt.close()


def plot_rms_for_alphas(errors, alphas):
    sns.set(style="ticks")
    plt.figure(figsize=(10, 5))
    colors = ['dodgerblue', 'mediumvioletred', 'olivedrab', 'darkcyan']
    linestyles = [':', '-', '--', '-.']
    for i, alpha in enumerate(alphas):
        plt.plot(errors[alpha], label='alpha={}'.format(str(alpha)), color=colors[i], linestyle=linestyles[i], linewidth=0.9)
    plt.xlabel('Walks/Episodes')
    plt.ylabel('RMS Error - averaged over states')
    # plt.yticks(np.arange(0.19, 0.36, 0.01))
    plt.title(r'Effect of different $\alpha $ values in TD(0) learning')
    plt.ylim(0.0, 0.3)
    plt.legend(loc="best")
    plt.savefig("output/random_walk_alphas_ex62.png")
    plt.close()


if __name__ == '__main__':
    # setting a seed to make the generating repeatable
    np.random.seed(15200)
    training_episodes = generate_training_episodes(n=1000)
    pred_values, errors = run_example62_values(training_episodes, alpha=0.1)  # alpha = 0.1 used from the book
    plot_value_graph(pred_values)

    error_alphas = {}
    alphas = [0.001, 0.004, 0.05, 0.1]
    for alpha in alphas:
        _, error_alphas[alpha] = run_example62_values(training_episodes, alpha=alpha)

    print(error_alphas)
    plot_rms_for_alphas(error_alphas, alphas)
