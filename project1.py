import json
import pickle
import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error

# Switches
USE_SAVED_TRAIN_DATA = True
PERFORM_MAIN_RUN = False
PERFORM_ADD_RUN = False

# 1. 100 training sets,each consisting of 10 sequence.
SEQ_COUNT = 10

# plot common
COLORS = ['mediumvioletred', 'dodgerblue', 'olivedrab', 'darkcyan']
MARKERS = ['o', 'x', '+', '>']
LINE_STYLES = ['-.', ':', '--', '--']


def execute_sequence(steps, weights, lamda, alpha, verbose=False):
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
            # print('delta_w_pk: ', delta_w_pk)
            # print('delta_p: ', delta_p)
        # print('delta_w: ', delta_w)

    return delta_w


def td_lamda_expt1(lamda, alpha, training, decay_alpha=False):
    # True predications/weights for each of the states are given as  [1/6, 1/3, 1/2, 2/3, 5/6] for B,C,D,E,F
    ideal_predictions = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 5.0 / 6.0]
    errors = []

    for i, sequences_set in enumerate(training):
        # components of the weight vector were initially set to 0.5 to prevent bias
        curr_weights = [0.5] * 5
        delta_w = [0.0] * 5
        not_converged = True
        # Each training set was presented repeatedly to each learning procedure
        # until the procedure no longer produced any significant changes in the weight vector
        while not_converged:
            # Repeat cyclically (1,2, .., 9,10, 1,2 ...) its ten sequences
            for sample_steps in sequences_set:
                delta_w = execute_sequence(sample_steps, curr_weights, lamda, alpha, verbose=(i == 4))
                # print('delta_w:', delta_w)
                # np.sum takes a longer time to converge
                max_dw = np.amax(np.absolute(delta_w))
                if max_dw < 0.001:
                    not_converged = False
                    # print('converged break: ', i)
                    break
            # delta_w's were accumulated over sequences and only used to update the weight vector
            # after the complete presentation of the training set
            curr_weights = np.add(curr_weights, delta_w)

        # if i % 25 == 0:
        #     print('training_sample : {}/100 completed'.format(i))
        errors.append(sqrt(mean_squared_error(curr_weights, ideal_predictions)))
        if decay_alpha:
            alpha = alpha * 0.999

    # print('size: {} lamda:- {} standard error: {:0.6f}'.format(len(training), lamda, stats.sem(errors)))
    return stats.sem(errors), np.mean(errors)


def td_lamda_expt2(lamda, alpha, training, decay_alpha=False):
    # True predications/weights for each of the states are given as  [1/6, 1/3, 1/2, 2/3, 5/6] for B,C,D,E,F
    ideal_predictions = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 5.0 / 6.0]
    errors = []

    for i, sequences_set in enumerate(training):
        # components of the weight vector were initially set to 0.5 to prevent bias
        curr_weights = [0.5] * 5
        # each training set was presentedonce to each procedure. No convergence check.
        for sample_steps in sequences_set:
            delta_w = execute_sequence(sample_steps, curr_weights, lamda, alpha)
            # print('delta_w:', delta_w)
            # weight updates were performed after each sequence
            curr_weights = np.add(curr_weights, delta_w)
            # print('curr_weights:', curr_weights)
        # if i % 50 == 0:
        #     print('training_sample : {}/100 completed'.format(i))
        errors.append(sqrt(mean_squared_error(curr_weights, ideal_predictions)))
        if decay_alpha:
            alpha = alpha * 0.995
    return np.mean(errors)


def generate_training_data(n_training_sets=100, terminate_early=True, load_file=True, file_name=None):
    # 2. vectors{xi} were the unit basis vectors of length 5, that is,
    # four of their components were 0 and the fifth was 1 (e.g., XD = (0,0,1,0,0)T)
    # https://www.techcoil.com/blog/how-to-save-and-load-objects-to-and-from-file-in-python-via-facilities-from-the-pickle-module/
    if load_file and file_name is not None:
        with open(file_name, 'rb') as training_data_file:
            print('Loading saved data - {}'.format(file_name))
            return pickle.load(training_data_file)
    else:
        training_sets = []

        for i in range(n_training_sets):
            sequence_set = []
            for j in range(SEQ_COUNT):
                # Every walk begins in the center state D.
                vector_xi = np.array([0, 0, 0, 1, 0, 0, 0])
                current_state = 3
                curr_seq_count = 1
                MAX_SEQ_LENGTH = 20
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
                    curr_seq_count += 1
                    # print("current_state: {}".format(current_state))
                    # print("seq.complete")
                    if terminate_early and curr_seq_count > MAX_SEQ_LENGTH:
                        break
                sequence_set.append(vector_xi)

            training_sets.append(sequence_set)

        with open(file_name, 'wb') as training_data_file:
            pickle.dump(training_sets, training_data_file)

        return training_sets


def experiment_1(lamda_values, training_data, decay_alpha=False):
    print(f'Experiment-1 with decay={decay_alpha}')
    results = {}
    std_error_results = {}
    start_time = time.time()
    for lamda in lamda_values:
        std_error_results[lamda], results[lamda] = td_lamda_expt1(lamda, 0.0044, training_data, decay_alpha)
        # print("lamda: {} - Error: {}".format(lamda, results[lamda]))
    time_taken = time.time() - start_time
    print(f'expt1: {json.dumps(results)}')
    print(f'std_error: {json.dumps(std_error_results)}')
    return time_taken, results, std_error_results


def experiment_2(training_data, decay_alpha=False):
    print('Experiment-2')
    results = {}
    best_alpha_dict = {}
    alpha_values = np.linspace(0, 0.6, 13)
    lamda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i, lamda in enumerate(lamda_values):
        results[lamda] = {}
        for alpha in alpha_values:
            # print("lamda: {}, alpha : {}".format(lamda, alpha))
            results[lamda][alpha] = td_lamda_expt2(lamda, alpha, training_data, decay_alpha=decay_alpha)

        best_alpha_dict[lamda] = get_min_key(results[lamda])
        # print('results[lamda]: ', results[lamda])

    print(json.dumps(results))
    print('best_alpha_dict: ', json.dumps(best_alpha_dict))
    return results, best_alpha_dict


def get_min_key(d):
    # https://stackoverflow.com/questions/3282823/get-the-key-corresponding-to-the-minimum-value-within-a-dictionary
    return min(d, key=lambda k: d[k])


def experiment_3(training_data, lamda_alpha_dict, decay_alpha=False):
    # Figure 5 plots the best error level achieved for each lamda value
    # Using the alpha value that was best for that lamda value.
    # As in the repeated-presentation experiment (similar to expt1)
    print('Experiment-3')
    results = {}
    lamda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for lamda in lamda_values:
        alpha = lamda_alpha_dict[lamda]
        # print("lamda: {} using alpha : {}".format(lamda, alpha))
        results[lamda] = td_lamda_expt2(lamda, alpha, training_data, decay_alpha)
        # print(results)
    print(json.dumps(results))
    return results


def generate_figure_3(results, file_name, lamda_values):
    sns.set(style="ticks")
    plt.figure(figsize=(6, 6))
    # https://stackoverflow.com/questions/18453566/python-dictionary-get-list-of-values-for-list-of-keys
    plot_results = {x: round(v, 3) for x, v in results.items() if x in lamda_values}

    # https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary
    plt.plot(*zip(*plot_results.items()), marker='o', color='mediumvioletred', linewidth=0.9)
    plt.xlabel(r'$\lambda $ values')
    plt.ylabel('ERROR')
    plt.ylim(0.17, 0.29)
    plt.xticks(lamda_values)

    plt.yticks(np.arange(0.19, 0.29, 0.01))
    plt.title('Figure 3- Average Error on the random-walk problem \n under repeated presentations.')
    plt.annotate('Widrow-Hoff', (0.7, 0.28))
    plt.savefig("output/{}.png".format(file_name))
    plt.close()


def generate_figure_4(results, file_name):
    sns.set(style="ticks")
    plt.figure(figsize=(7, 6))
    plt.xlabel(r'$\alpha $ values')
    plt.ylabel('ERROR')

    lamdas_to_plot = [0.0, 0.3, 0.8, 1.0]
    for i, lamda in enumerate(lamdas_to_plot):
        results_lamda = convert_keys(results[lamda])
        label = r'$\lambda $= 1 (Widrow-Hoff)' if lamda == 1 else r'$\lambda $= ' + str(lamda)
        plt.plot(*zip(*results_lamda.items()), marker='o', color=COLORS[i], label=label, linewidth=0.9)
    plt.title('Figure 4 - Average Error on random-walk problem \n after experiencing 10 sequences')
    # plt.xlim(0.0, 0.6)
    plt.ylim(0.05, 0.75)
    plt.legend()
    plt.savefig("output/{}.png".format(file_name))
    plt.close()


def generate_figure_5(results, file_name):
    lamda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sns.set(style="ticks")
    plt.figure(figsize=(7, 7))

    plot_results = {x: round(v, 3) for x, v in results.items() if x in lamda_values}

    # https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary
    plt.plot(*zip(*plot_results.items()), marker='o', color='dodgerblue', linewidth=0.9)
    plt.xlabel(r'$\lambda $ values')
    plt.ylabel(r'ERROR using best $\alpha $')
    plt.xticks(lamda_values)
    plt.ylim(0.09, 0.21)
    plt.yticks(np.arange(0.1, 0.22, 0.02))
    # plt.axhline(y=0.097, color='r', linestyle='--')
    plt.title(r'Figure 3 - Average Error at best $\alpha $ value on random-walk problem.')
    plt.annotate('Widrow-Hoff', (0.75, 0.19))
    plt.savefig("output/{}.png".format(file_name))
    plt.close()


def convert_keys(dict):
    # https://stackoverflow.com/questions/12117080/how-do-i-create-dictionary-from-another-dictionary
    return {round(float(x), 2): dict[x] for x in dict.keys()}


def generate_ideal_comparison_expt1(tag_type):
    with open('results.json', 'r') as f:
        results = json.load(f)

    sns.set(style="ticks")
    plt.figure(figsize=(7, 6))
    lamda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    ideal_values = convert_keys(results['ideal']['expt1'])
    results_non_decay = convert_keys(results[tag_type]['expt1'])
    results_decay = convert_keys(results[f'{tag_type}_decay']['expt1'])

    plot_results = {x: round(v, 4) for x, v in results_non_decay.items() if x in lamda_values}
    plt.plot(*zip(*plot_results.items()), marker='o', color='mediumvioletred', linewidth=0.9, linestyle='-', label='Reproduction results')

    plot_results = {x: round(v, 4) for x, v in results_decay.items() if x in lamda_values}
    plt.plot(*zip(*plot_results.items()), marker='>', color='olivedrab', linewidth=0.9, linestyle='-', label=r'Reproduction results - $\alpha $ decay')

    plt.plot(*zip(*ideal_values.items()), marker='x', color='darkcyan', linewidth=0.9, linestyle=':', label='Sutton results')
    plt.xlabel(r'$\lambda $ values')
    plt.ylabel('ERROR')
    plt.ylim(0.17, 0.29)
    plt.xticks(lamda_values)

    plt.yticks(np.arange(0.19, 0.29, 0.01))
    plt.title('Average Error on the random-walk problem \n comparison with Sutton results')
    plt.legend()
    plt.savefig(f"output/expt_1_{tag_type}_sutton_comparison.png")
    plt.close()


def plot_rmse(ep_lengths, multi_train_results):
    with open('results.json', 'r') as f:
        results = json.load(f)

    sns.set(style="ticks")
    plt.figure(figsize=(6, 6))
    plt.xlabel(r'$\lambda $ values')
    plt.ylabel('RMSE - error')
    for i, ep_length in enumerate(ep_lengths):
        plt.plot(*zip(*multi_train_results[ep_length].items()), marker=MARKERS[i], color=COLORS[i], label=f'episodes {str(ep_length)}', linestyle=LINE_STYLES[i], linewidth=0.9, alpha=0.8)

    orig_results = results['short']['expt1']
    plt.plot(*zip(*orig_results.items()), marker=MARKERS[3], color=COLORS[3], label=f'Original results(100 episodes)', linestyle='-')

    plt.title('Average Error on the random-walk problem \n under repeated presentations for different training set sizes')
    plt.legend()
    plt.savefig("output/train_set_sizes_expt1.png")
    plt.close()


def run_sample_cases(lamda, alpha=0.1):
    training_set = [['E', 'F', 'G']]
    #
    # ['C', 'B', 'A'],
    # ['C', 'D', 'E', 'D', 'E', 'F', 'G']

    state_pos = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    np_set = []
    for seq in training_set:
        vector_xi = np.array([0, 0, 0, 1, 0, 0, 0])
        for state in seq:
            new_state = np.array([0, 0, 0, 0, 0, 0, 0])
            new_state[state_pos.index(state)] = 1
            vector_xi = np.vstack([vector_xi, new_state])
        np_set.append(vector_xi)
    td_lamda_expt2(lamda, alpha, [np_set])


def run_multiple_training_sets(lamda_values, perform_run=False):
    print('Running multiple training sets')
    multi_train_results = {}
    ep_lengths = [50, 200, 500]

    if perform_run:
        for n in ep_lengths:
            training_data = generate_training_data(n_training_sets=n, terminate_early=True, load_file=USE_SAVED_TRAIN_DATA, file_name='files/training_set_{}.data'.format(n))
            _, results, _ = experiment_1(lamda_values, training_data, decay_alpha=True)
            multi_train_results[n] = results
        print('multi_train_results: ', json.dumps(multi_train_results))
    else:
        with open('results.json', 'r') as f:
            results = json.load(f)
        multi_train_results = convert_keys(results['multi_train_results'])

    plot_rmse(ep_lengths, multi_train_results)


def generate_only_term_episodes(n_training_sets=100, load_file=True, file_name=None, max_seq_length=20):
    # 2. vectors{xi} were the unit basis vectors of length 5, that is,
    # four of their components were 0 and the fifth was 1 (e.g., XD = (0,0,1,0,0)T)
    # https://www.techcoil.com/blog/how-to-save-and-load-objects-to-and-from-file-in-python-via-facilities-from-the-pickle-module/
    if load_file and file_name is not None:
        with open(file_name, 'rb') as training_data_file:
            print('Loading saved data - {}'.format(file_name))
            return pickle.load(training_data_file)
    else:
        training_sets = []

        for i in range(n_training_sets):
            sequence_set = []
            j = 0
            while j < SEQ_COUNT:
                # Every walk begins in the center state D.
                vector_xi = np.array([0, 0, 0, 1, 0, 0, 0])
                current_state = 3
                curr_seq_count = 1
                early_termination = False
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
                    curr_seq_count += 1
                    # print("current_state: {}".format(current_state))
                    # print("seq.complete")
                    if curr_seq_count > max_seq_length:
                        early_termination = True
                        break

                if not early_termination:
                    sequence_set.append(vector_xi)
                    j += 1

            training_sets.append(sequence_set)

        with open(file_name, 'wb') as training_data_file:
            pickle.dump(training_sets, training_data_file)

        return training_sets


def run_multiple_episode_lengths(lamda_values, perform_run=False):
    results_term = {}
    episode_lengths = [10, 15, 25]

    if perform_run:
        for seq_length in episode_lengths:
            training_data = generate_only_term_episodes(load_file=False, file_name='files/training_set_term_ep_{}.data'.format(seq_length), max_seq_length=seq_length)
            _, results_term[seq_length], _ = experiment_1(lamda_values, training_data, decay_alpha=True)

        print(f'only_term: {json.dumps(results_term)}')
    else:
        with open('results.json', 'r') as f:
            results = json.load(f)
        results_term = convert_keys(results['term_ep'])

    generate_expt1_comparison(episode_lengths, results_term)


def generate_expt1_comparison(episode_lengths, results_term):
    with open('results.json', 'r') as f:
        results = json.load(f)
    orig_results = convert_keys(results['short']['expt1'])

    sns.set(style="ticks")
    plt.figure(figsize=(6, 6))
    lamda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for i, ep_length in enumerate(episode_lengths):
        plot_results = {x: round(v, 3) for x, v in results_term[ep_length].items() if float(x) in lamda_values}
        plt.plot(*zip(*plot_results.items()), marker='o', color=COLORS[i], linestyle=LINE_STYLES[i], linewidth=0.9, label='limit={}'.format(ep_length))

    plot_results = {str(x): round(v, 3) for x, v in orig_results.items() if float(x) in lamda_values}
    plt.plot(*zip(*plot_results.items()), marker='x', color='darkcyan', linestyle='-', linewidth=0.9, label='reproduction-results (limit=20)')
    plt.xlabel(r'$\lambda $ values')
    plt.ylabel('RMS-Error')
    plt.title('Average Error on the random-walk problem \n under repeated presentations for various episode lengths.')
    plt.legend(loc="best")
    plt.savefig("output/term_ep_comp.png")
    plt.close()


def generate_std_error_graph(results, tag):
    sns.set(style="ticks")
    plt.xlabel(r'$\lambda $ values')
    plt.ylim(0.0001, 0.012)
    plt.ylabel('ERROR')
    plot_results = {x: round(float(v), 5) for x, v in results.items()}
    plt.plot(*zip(*plot_results.items()), marker='x', color='dodgerblue', label='standard error', linestyle='-', linewidth=0.9)
    plt.title('Standard Error for RMSE on the random-walk problem \n after repeated presentations')
    plt.axhline(y=0.01, color='darkcyan', linestyle=':', linewidth=0.8)
    plt.legend(loc="best")
    plt.savefig(f"output/{tag}_std_error_plot.png")
    plt.close()


def run_sequence(expt1_lamda_values, file_name, tag, decay_alpha=False, perform_run=True):
    print('Running short sequences')
    with open('results.json', 'r') as f:
        results = json.load(f)

    if perform_run:
        # You will be replicating figures 3, 4, and 5 (Check Erratum at the end of paper)
        training_data = generate_training_data(terminate_early=True, load_file=USE_SAVED_TRAIN_DATA, file_name=file_name)

        _, results_expt1, std_error = experiment_1(expt1_lamda_values, training_data, decay_alpha=decay_alpha)
        results_expt2, best_alpha_dict = experiment_2(training_data, decay_alpha=decay_alpha)
        results_expt3 = experiment_3(training_data, best_alpha_dict, decay_alpha=decay_alpha)
    else:
        results_expt1 = convert_keys(results[tag]['expt1'])
        std_error = convert_keys(results[tag]['std_error'])
        results_expt2 = convert_keys(results[tag]['expt2'])
        best_alpha_dict = convert_keys(results[tag]['best_alpha_dict'])
        results_expt3 = convert_keys(results[tag]['expt3'])

    for k, v in best_alpha_dict.items():
        if k in expt1_lamda_values:
            print('best alpha for {} is {:0.2f}'.format(k, v))

    generate_figure_3(results_expt1, f'{tag}_expt_1', expt1_lamda_values)
    generate_std_error_graph(std_error, tag)
    generate_figure_4(results_expt2, f'{tag}_expt_2')
    generate_figure_5(results_expt3, f'{tag}_expt_3')


def generate_ideal_comparison_expt3(tag_type):
    with open('results.json', 'r') as f:
        results = json.load(f)

    sns.set(style="ticks")
    plt.figure(figsize=(7, 6))
    lamda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    ideal_values = convert_keys(results['ideal']['expt3'])
    results_non_decay = convert_keys(results[tag_type]['expt3'])
    results_decay = convert_keys(results[f'{tag_type}_decay']['expt3'])

    plot_results = {x: round(v, 4) for x, v in results_non_decay.items() if x in lamda_values}
    plt.plot(*zip(*plot_results.items()), marker='o', color='mediumvioletred', linewidth=0.9, linestyle='-', label='Reproduction results')

    plot_results = {x: round(v, 4) for x, v in results_decay.items() if x in lamda_values}
    plt.plot(*zip(*plot_results.items()), marker='>', color='olivedrab', linewidth=0.9, linestyle='-', label=r'Reproduction results - $\alpha $ decay')

    plt.plot(*zip(*ideal_values.items()), marker='x', color='darkcyan', linewidth=0.9, linestyle=':', label='Sutton results')
    plt.xlabel(r'$\lambda $ values')
    plt.ylabel('ERROR')
    # plt.ylim(0.17, 0.29)
    # plt.yticks(np.arange(0.19, 0.29, 0.01))
    plt.xticks(lamda_values)
    plt.title('Average Error on the random-walk problem \n comparison with Sutton results')
    plt.legend()
    plt.savefig(f"output/expt_3_{tag_type}_sutton_comparison.png")
    plt.close()


if __name__ == '__main__':
    # setting a seed to make the generating repeatable
    np.random.seed(15200)
    expt1_lamda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    run_sequence(expt1_lamda_values, file_name='files/short_sequences.data', tag='short', decay_alpha=False, perform_run=PERFORM_MAIN_RUN)
    run_sequence(expt1_lamda_values, file_name='files/short_sequences.data', tag='short_decay', decay_alpha=True, perform_run=PERFORM_MAIN_RUN)

    generate_ideal_comparison_expt1(tag_type='short')
    generate_ideal_comparison_expt3(tag_type='short')

    run_multiple_training_sets(expt1_lamda_values, perform_run=PERFORM_ADD_RUN)
    run_multiple_episode_lengths(expt1_lamda_values, perform_run=PERFORM_ADD_RUN)

    run_sample_cases(lamda=0)
    run_sample_cases(lamda=0.3)

    # Ignore these method calls
    # run_sequence(expt1_lamda_values, file_name='files/full_sequences.data', tag='full', decay_alpha=False, perform_run=True)
    # run_sequence(expt1_lamda_values, file_name='files/full_sequences.data', tag='full_decay', decay_alpha=True, perform_run=True)
