import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import itertools


def create_param_dict(neuron_config):
    param_list = []
    parameters = {"El": [-0.1, -0.5, 0, 0.5, 0.1], "dt": [1e-5, 3e-5, 5e-5, 7e-5, 9e-5],
                  "R_input": [0.01834e9, 0.05834e9, 0.1834e9, 0.5834e9, 1.834e9],
                  "C": [0.01e-9, 0.05e-9, 0.1e-9, 0.5e-9, 1e-9],
                  "th_inf": [0.00541, 0.0141, 0.0341, 0.0741, 0.141]}

    for p, values in sorted(parameters.items()):
        param_list.append(values)

    comb_list = list(itertools.product(*param_list))
    params = []
    for id, comb in enumerate(comb_list):
        neuron_config["C"] = comb[0]
        neuron_config["El"] = comb[1]
        neuron_config["R_input"] = comb[2]
        neuron_config["dt"] = comb[3]
        neuron_config["th_inf"] = comb[4]

        params.append(neuron_config.copy())

    return params


def loss_fn(original_params, predicted_params):
    mse_weight = 1

    mse = F.mse_loss(predicted_params, original_params, reduction='mean')

    mse = mse * mse_weight

    return mse


def plot_trace(cfg, neuron, stimulus, output, spike_times, stim, config):
    signal_allow = cfg.signal_allow
    voltage = output['voltage'][: signal_allow]
    threshold = output['threshold'][: signal_allow]

    time = np.arange(1, signal_allow + 1, step=1) * neuron.dt

    plt.figure(figsize=(10, 10))

    # plot stimulus
    plt.subplot(3, 1, 1)
    plt.plot(time, stimulus)
    plt.xlabel('time (s)')
    plt.ylabel('current (A)')
    plt.title('Stimulus : ' + str(stim))

    # plot model output
    plt.subplot(3, 1, 2)
    plt.plot(time, voltage, label='voltage')
    plt.plot(time, threshold, label='threshold')
    plt.xlabel('time (s)')
    plt.ylabel('voltage (V)')
    plt.legend(loc=3)
    plt.title('Model Response : ' + str(config))

    # plot spike times
    plt.subplot(3, 1, 3)
    if len(spike_times) != 0:
        plt.stem(spike_times, np.ones((len(spike_times, ))))

    plt.xlabel('time (s)')
    plt.title('Stem of Spike Times')

    plt.tight_layout()

    plt.savefig(config + '_' + stim + '.png')

    return plt
