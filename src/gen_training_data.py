import allensdk.core.json_utilities as json_utilities
from src.config import Config
from allensdk.model.glif.glif_neuron import GlifNeuron
from allensdk.core.nwb_data_set import NwbDataSet
from src.utils import *

cfg = Config()

data_dir = cfg.data_dir
ephys_file_name = cfg.ephys_file_name

neuron_config = cfg.neuron_config
neuron_id = next(iter(neuron_config))
neuron_config = neuron_config[neuron_id]
ephys_sweeps = cfg.ephys_sweeps

stim_names = ["Noise 1"]
params = create_param_dict(neuron_config)

for stim in stim_names:

    sweeps = [s for s in ephys_sweeps
              if s['stimulus_name'] == stim]

    for sweep_id, ephys_sweep in enumerate(sweeps):
        ds = NwbDataSet(ephys_file_name)
        data = ds.get_sweep(ephys_sweep['sweep_number'])
        stimulus = data['stimulus']
        stimulus = stimulus[stimulus != 0]
        stimulus = stimulus[:cfg.stimulus_allow]

        for param_id, param_dict in enumerate(params):
            train = {}

            # update neuron_config based on params
            neuron_config = param_dict

            # initialize the neuron
            neuron = GlifNeuron.from_dict(neuron_config)

            # Set dt
            neuron.dt = 1.0 / data['sampling_rate']

            # simulate the neuron
            output = neuron.run(stimulus)

            voltage = output['voltage'] * 1e3

            if stim == "Noise 1":
                save = 'n1'
            else:
                save = 'n2'

            train["voltage"] = voltage
            train["params"] = param_dict

            np.save(data_dir + save + '_' + str(sweep_id) + '_' + str(param_id) + '.npy',
                    train)

if __name__ == "__main__":
    pass
