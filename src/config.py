import allensdk.core.json_utilities as json_utilities
import torch


class Config:
    data_dir = "/data2/neuro/data/train"
    ephys_file_name = '../files/neuron/stimulus.nwb'

    neuron_config = json_utilities.read('../files/neuron/neuron_config.json')
    ephys_sweeps = json_utilities.read('../files/neuron/ephys_sweeps.json')
    fit_config = "../files/configs/fit_config.npy"
    nn_config = "../files/configs/nn_config.npy"
    exp_config = "../files/configs/exp_config.npy"

    neuron_id = "485513169"
    signal_allow = 50000
    stimulus_allow = 52000

    epochs = 2
    learning_rate = 1e-2
    device = torch.device('cuda:0')

    input_dim = 1
    hidden_dim = 5
    param_dim = 5


