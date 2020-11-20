from functools import partial
import torch
import torch.utils.data
import numpy as np


try:
    from models import SLNetwork
except:
    from .models import SLNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simulations_file = 'states.npy'
actions_file = 'actions.npy'
BATCH_SIZE = 10
NUM_EPOCHS = 100
NUM_ACTIONS = 5

if float(torch.version.__version__[:3]) >= 1.6:
    torch.save = partial(torch.save, _use_new_zipfile_serialization=False)


class SimulationDataset(torch.utils.data.TensorDataset):

    def __init__(self, x_npy_file, y_npy_file):
        X, y = self.load_numpy_data(x_npy_file, y_npy_file)
        super().__init__(X, y)

    def load_numpy_data(self, x_file, y_file):
        with open(x_file, 'rb') as f:
            x = np.load(f)
        with open(y_file, 'rb') as f:
            y = np.load(f)
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
        return x, y

loader = torch.utils.data.DataLoader(
    dataset=SimulationDataset(simulations_file, actions_file),
    batch_size=BATCH_SIZE, shuffle=True
)


def train(model, start_epoch=0, num_epochs=NUM_EPOCHS, save_dir=''):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    if start_epoch > 0:
        model.load_state_dict(torch.load(save_dir + "/epoch-" + str(start_epoch) + ".model"))
        optimizer.load_state_dict(torch.load(save_dir + "/epoch-" + str(start_epoch) + ".opt"))

    last_epoch = start_epoch
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0

        for i, batch in enumerate(loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            prediction = model(x)
            loss = loss_func(prediction, y)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss /= len(loader.dataset)
        torch.save(model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
        torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
        last_epoch = epoch
        print("[epoch %d]: loss = %f" % (epoch + 1, epoch_loss))

    return model, last_epoch + 1

def run():
    model = SLNetwork(loader.dataset[0][0].shape, NUM_ACTIONS)
    train(model, save_dir='weights', start_epoch=0, num_epochs=20)
