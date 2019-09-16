import pandas as pd
import logging
import numpy as np
from os import listdir
from os.path import isfile, join
import traceback
import csv

logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None


class DataPrep:

    def __init__(self, mode, cfg):
        self.train_data_path = "/data2/neuro/data/train/"
        self.valid_data_path = "/data2/neuro/data/valid/"
        self.test_data_path = "/data2/neuro/data/test/"

        if mode == "train":
            self.data_path = self.train_data_path
        elif mode == "valid":
            self.data_path = self.valid_data_path
        else:
            self.data_path = self.test_data_path

        self.mode = mode
        self.signal_allow = cfg.signal_allow

    def get_data(self):

        signal_files = [f for f in listdir(self.data_path) if isfile(join(self.data_path, f))]

        for id, file in enumerate(signal_files):
            data = np.load(self.data_path + file).item()

            signal = data["voltage"]
            params = data["params"]
            params = self.convert_params(params)

            signal = signal[~np.isnan(signal)]
            signal = signal[:self.signal_allow]
            params = self.convert_dict_to_list(params)

            try:
                yield signal, params
            except Exception as e:
                logger.error(traceback.format_exc())

    def convert_params(self, params):

        params["C"] = params["C"] * 1e9
        params["R_input"] = params["R_input"] * 1e-9
        params["dt"] = params["dt"] * 1e5

        return params

    def convert_dict_to_list(self, params):

        param_list = []
        parameters = ["C", "El", "R_input", "dt", "th_inf"]
        for p in parameters:
            param_list.append(params[p])

        return np.array(param_list)

    def get_file_signal(self, file_path):
        with open(file_path, newline='') as csvfile:
            signal = np.array(list(csv.reader(csvfile)))[0]
            signal = np.array([float(i) for i in signal])
            signal = np.expand_dims(signal[:self.signal_allow], axis=0)

        return signal


if __name__ == '__main__':
    data_ob_vae = DataPrep(mode='train')

    data_gen = data_ob_vae.get_data()
    cut_seq = next(data_gen)
