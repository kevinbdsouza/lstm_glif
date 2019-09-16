import os
import torch
import torch.utils.data
import torch.optim as optim
from src.ml_model import SeqParamModel
from src.utils import *
from src.data_prep import DataPrep
from src.post_train import PostTrain
from src.config import Config


class Trainer(object):
    def __init__(self, cfg, model, epochs=2, learning_rate=1e-4, checkpoints='model.pth',
                 device=torch.device('cuda:0')):

        self.start_epoch = 0
        self.cfg = cfg
        self.epochs = epochs
        self.device = device
        self.model = model
        self.model.to(device)
        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.data_ob_train = DataPrep(mode='train', cfg=cfg)
        self.data_ob_valid = DataPrep(mode='valid', cfg=cfg)
        self.data_ob_test = DataPrep(mode='test', cfg=cfg)
        self.cur_dir = "/home/kevindsouza/Documents/projects/neuro/src/"
        self.model_save_dir = "/home/kevindsouza/Documents/projects/neuro/model/"

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.train_losses},
            self.checkpoints)

    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def single_eval(self, input):

        input = torch.from_numpy(input).float().to(self.device)
        input = torch.unsqueeze(torch.unsqueeze(input, 0), 2)

        predicted_params = self.model(input)
        predicted_params = torch.squeeze(torch.squeeze(predicted_params, 0), 0)
        predicted_params = predicted_params.cpu().detach().numpy()

        return predicted_params

    def test_run(self, datagen_test):
        for test_input, test_params in datagen_test:
            test_input = torch.from_numpy(test_input).float().to(self.device)
            test_input = torch.unsqueeze(torch.unsqueeze(test_input, 0), 2)
            test_params = torch.from_numpy(test_params).float().to(self.device)

            predicted_params = self.model(test_input)

            predicted_params = torch.squeeze(torch.squeeze(predicted_params, 0), 0)
            test_loss = loss_fn(test_params, predicted_params)
            self.test_losses.append(test_loss.item())

        return np.mean(self.test_losses)

    def valid_run(self, datagen_valid):
        valid_losses = []
        for valid_input, valid_params in datagen_valid:
            valid_input = torch.from_numpy(valid_input).float().to(self.device)
            valid_input = torch.unsqueeze(torch.unsqueeze(valid_input, 0), 2)
            valid_params = torch.from_numpy(valid_params).float().to(self.device)

            predicted_params = self.model(valid_input)

            predicted_params = torch.squeeze(torch.squeeze(predicted_params, 0), 0)
            valid_loss = loss_fn(valid_params, predicted_params)
            valid_losses.append(valid_loss.item())

        return np.mean(valid_losses)

    def train_model(self):
        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            print("Running Epoch : {}".format(epoch + 1))

            datagen_train = self.data_ob_train.get_data()
            datagen_valid = self.data_ob_valid.get_data()

            count = 0
            for train_input, train_params in datagen_train:
                train_input = torch.from_numpy(train_input).float().to(self.device)
                train_input = torch.unsqueeze(torch.unsqueeze(train_input, 0), 2)
                train_params = torch.from_numpy(train_params).float().to(self.device)

                self.optimizer.zero_grad()
                predicted_params = self.model(train_input)

                predicted_params = torch.squeeze(torch.squeeze(predicted_params, 0), 0)
                train_loss = loss_fn(train_params, predicted_params)

                train_loss.backward()
                self.optimizer.step()
                losses.append(train_loss.item())
                print("Iter : {} Loss : {}".format(count, train_loss))
                count += 1

            meanloss = np.mean(losses)
            self.train_losses.append(meanloss)
            print("Epoch {} : Average Loss: {}".format(epoch + 1, meanloss))

            self.save_checkpoint(epoch)

            os.system("cp {}{} {}".format(self.cur_dir, "model.pth", self.model_save_dir))

            self.model.eval()

            valid_loss = self.valid_run(datagen_valid)
            self.valid_losses.append(valid_loss)

            self.model.train()

        print("Training is complete")

    def test_model(self):

        datagen_test = self.data_ob_test.get_data()
        self.model.eval()

        test_loss = self.test_run(datagen_test)

        return test_loss

    def get_glif_params(self, input):

        glif_params = self.single_eval(input)

        return glif_params


if __name__ == "__main__":
    cfg = Config()

    epochs = cfg.epochs
    learning_rate = cfg.learning_rate
    device = cfg.device

    model = SeqParamModel(cfg)

    trainer = Trainer(cfg, model, epochs=epochs, learning_rate=learning_rate, device=device)

    trainer.load_checkpoint()

    # trainer.train_model()
    # test_loss = trainer.test_model()

    post_ob = PostTrain(cfg)
    """
    _, voltage, _, _ = post_ob.get_voltage(post_ob.neuron_config, "Noise 2")
    glif_params = trainer.get_glif_params(voltage)
    post_ob.create_nn_config(glif_params)
    post_ob.get_post_traces()
    """
    post_ob.get_random_config()
