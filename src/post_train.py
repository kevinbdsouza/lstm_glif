import allensdk.core.json_utilities as json_utilities
from allensdk.model.glif.glif_neuron import GlifNeuron
from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.ephys.ephys_features import detect_putative_spikes
from src.utils import *
from random import randint
import matplotlib.pyplot as plt


class PostTrain:
    def __init__(self, cfg):
        self.ephys_file_name = cfg.ephys_file_name
        self.neuron_config = np.load(cfg.fit_config).item()
        self.neuron_id = cfg.neuron_id
        self.cfg = cfg

    def get_voltage(self, neuron_config, stim_name):

        ephys_sweeps = self.cfg.ephys_sweeps

        ephys_sweep = next(s for s in ephys_sweeps
                           if s['stimulus_name'] == stim_name)

        ds = NwbDataSet(self.ephys_file_name)
        data = ds.get_sweep(ephys_sweep['sweep_number'])
        stimulus = data['stimulus']
        stimulus = stimulus[stimulus != 0]
        stimulus = stimulus[:self.cfg.stimulus_allow]

        # initialize the neuron
        neuron = GlifNeuron.from_dict(neuron_config)

        # Set dt
        neuron.dt = 1.0 / data['sampling_rate']

        # simulate the neuron
        output = neuron.run(stimulus)

        voltage = output['voltage'] * 1e3

        voltage = voltage[~np.isnan(voltage)]
        voltage = voltage[:self.cfg.signal_allow]

        return output, voltage, neuron, stimulus

    def plot_loss(self):
        train_loss = "train_losses.npy"
        valid_loss = "valid_losses.npy"

        epochs = range(1, 6)
        tr_loss = np.load(train_loss)
        vl_loss = np.load(valid_loss)

        plt.xticks(np.arange(1, 6, step=1))
        plt.plot(epochs, tr_loss, 'r', label="training loss")
        plt.plot(epochs, vl_loss, 'g', label="validation loss")
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Loss across epochs')
        plt.legend()
        plt.savefig("loss.png")

        return

    def create_nn_config(self, glif_params):
        glif_config = self.neuron_config

        glif_config["C"] = glif_params[0] * 1e-9
        glif_config["El"] = glif_params[1]
        glif_config["R_input"] = glif_params[2] * 1e9
        glif_config["dt"] = glif_params[3] * 1e-5
        glif_config["th_inf"] = glif_params[4]

        np.save('nn_config.npy', glif_config.copy())

    def get_random_config(self):
        params = create_param_dict(self.neuron_config)

        id = randint(0, len(params) - 1)
        exp_config = params[id]

        np.save('exp_config.npy', exp_config)

    def get_post_traces(self):
        stims = ["Noise 1", "Noise 2", "Ramp", "Long Square"]

        fit_config = np.load(self.cfg.fit_config).item()
        nn_config = np.load(self.cfg.nn_config).item()
        exp_config = np.load(self.cfg.exp_config).item()

        configs = {"fitted": fit_config, "nn": nn_config, "exp": exp_config}

        spike_counts = {}
        for stim in stims:
            for config_name, config in sorted(configs.items()):
                output, voltage, neuron, stimulus = self.get_voltage(config, stim)

                if config_name == "fitted":
                    cutoff = 0.1
                else:
                    cutoff = 0.005

                spike_times, spike_count = self.get_spike_counts(voltage, neuron, cutoff=cutoff)

                spike_times = spike_times * neuron.dt
                plt = plot_trace(self.cfg, neuron, stimulus[:self.cfg.signal_allow], output, spike_times, stim,
                                 config_name)

                plt.show()

                spike_counts[str(config_name + '_' + stim)] = spike_count

        np.save('spike_count_dict.npy', spike_counts)

    def get_spike_counts(self, voltage, neuron, cutoff):

        time = np.arange(1, 50001, step=1) * neuron.dt

        spike_indices = detect_putative_spikes(voltage, time, dv_cutoff=cutoff)
        spike_counts = len(spike_indices)

        return spike_indices, spike_counts


if __name__ == "__main__":
    test_voltage = None

    post_ob = PostTrain()
    # post_ob.plot_loss()

    spike_counts = np.load('spike_count_dict.npy').item()
