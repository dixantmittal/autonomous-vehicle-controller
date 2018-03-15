import _pickle
import argparse

import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset

import utils
from network_architectures import *

recurrent_control_net = 'recurrent_control_net'
control_net = 'control_net'

networks = {
    recurrent_control_net: RecurrentControlNet('model/recurrentcontrolnet.pt'),
    control_net: ControlNet('model/controlnet.pt')
}

hidden_state = None
previous_frames = []
previous_controls = []

optimizer = None

lr = 3e-6

use_gpu = torch.cuda.is_available()


class CustomDataSet(Dataset):
    def __init__(self, X, y, is_normalized=True):
        if not is_normalized:
            X = utils.normalize_image(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        X_batch = self.X[index]
        y_batch = self.y[index]

        return X_batch, y_batch

    def __len__(self):
        return len(self.X)


def forward_propagation(model, X, hidden_state=None):
    prediction, hidden_state = model(X, hidden_state)
    return prediction, hidden_state


def backward_propagation(optimizer, loss):
    # clean the optimizer
    optimizer.zero_grad()

    # backprop the gradient
    loss.backward()

    # update weights
    optimizer.step()

    return loss.data[0]


def train(model, X, y):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    cudnn.benchmark = True

    loss_fn = torch.nn.MSELoss()

    if use_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    dataloader = DataLoader(dataset=CustomDataSet(X, y),
                            batch_size=args.batch_size,
                            num_workers=10)

    loss_history = []

    print('#' * 10, 'Starting Training', '#' * 10, '\n')
    for epoch in range(args.epochs):
        for itr, batch in enumerate(dataloader):
            X = Variable(batch[0], requires_grad=False).type(torch.FloatTensor)
            y = Variable(batch[1], requires_grad=False).type(torch.FloatTensor)

            if use_gpu:
                X = X.cuda()
                y = y.cuda()

            # Forward pass
            prediction, _ = forward_propagation(model, X, None)

            # calculate the loss
            loss = loss_fn(prediction, y)

            # Back pass
            loss = backward_propagation(optimizer, loss)

            # collect loss
            loss_history.append(loss)
            print("Itr/Epoch: %2d/%2d loss: %.6f" % (itr, epoch, loss))

        # after each epoch, save the weights
        utils.save(model)
        np.save('loss_history.npy', np.array(loss_history))

    print('\n', '#' * 10, 'Training Finished', '#' * 10)


def inference(network, X, online_training=False, y=None):
    global hidden_state
    model = networks[network]

    if online_training:
        model.train()
        X = np.expand_dims(X, axis=0)
        X, y = modify_if_sequential(X, y, network)
        y = Variable(torch.from_numpy(np.array([y])), requires_grad=False).type(torch.FloatTensor)
        if use_gpu:
            y = y.cuda()

        hidden_state = None
    else:
        model.eval()

    X = Variable(torch.from_numpy(X), requires_grad=False).type(torch.FloatTensor)

    if use_gpu:
        model = model.cuda()
        X = X.cuda()

    prediction, hidden_state = forward_propagation(model, X, hidden_state)

    if online_training:
        loss_fn = torch.nn.MSELoss()
        if use_gpu:
            loss_fn = loss_fn.cuda()

        loss = loss_fn(prediction, y)

        if loss.data[0] < 35:
            backward_propagation(torch.optim.Adam(model.parameters(), lr=lr), loss)

    if hidden_state is not None:
        hidden_state = Variable(hidden_state.data)
        prediction = prediction.squeeze()
    return prediction[-1]


def modify_if_sequential(X, y, network):
    if network == recurrent_control_net:
        if len(previous_frames) >= 10:
            previous_frames.pop(0)
            previous_controls.pop(0)

        previous_frames.append(X)
        previous_controls.append(y)

        b, h, w, c = X.shape
        X = np.array(previous_frames).reshape(-1, h, w, c)
        y = np.array(previous_controls)

    return X, y


def get_model_object(network):
    return networks[network]


def init(network):
    global previous_controls
    global previous_frames
    previous_controls, previous_frames = [], []
    model = get_model_object(network)

    try:
        print()
        print('>> Loading Model... <<')
        print('> Model Path: ', model.model_path)
        model_state = torch.load(model.model_path)
        model.load_state_dict(model_state['state_dict'])
        print('> Model found and loaded!!')
    except (OSError, _pickle.UnpicklingError):
        print('> Model not found. Training from scratch.')
    print()

    return model


def sanity_check(network, X, y):
    global optimizer
    loss_fn = torch.nn.MSELoss()

    model = networks[network]
    if use_gpu:
        model = model.cuda()
    model.train()

    # start training
    dataloader = DataLoader(dataset=CustomDataSet(X[:100], y[:100, 0]),
                            batch_size=10,
                            num_workers=10)

    print('#' * 10, 'Overfitting Small Data', '#' * 10, '\n')
    for epoch in range(args.epochs):
        hidden = None
        for itr, batch in enumerate(dataloader):
            X = Variable(batch[0], requires_grad=False).type(torch.FloatTensor)
            y = Variable(batch[1], requires_grad=False).type(torch.FloatTensor)

            if use_gpu:
                X = X.cuda()
                y = y.cuda()

            prediction, hidden = model(X, hidden)
            if hidden is not None:
                hidden = Variable(hidden.data)

            optimizer.zero_grad()
            loss = loss_fn(prediction, y)

            loss.backward()

            optimizer.step()
            print("Epoch: %2d loss: %.6f" % (epoch, loss))

    print('\n', '#' * 10, 'Overfitting Done', '#' * 10)

    print('#' * 10, 'Finding Learning Rate', '#' * 10, '\n')

    for i in range(20):
        init(network)
        model = networks[network]
        if use_gpu:
            model = model.cuda()

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=10 ** np.random.uniform(-4, -1))
        for epoch in range(args.epochs // 10):
            hidden = None
            for itr, batch in enumerate(dataloader):
                X = Variable(batch[0], requires_grad=False).type(torch.FloatTensor)
                y = Variable(batch[1], requires_grad=False).type(torch.FloatTensor)

                if use_gpu:
                    X = X.cuda()
                    y = y.cuda()

                prediction, hidden = model(X, hidden)
                if hidden is not None:
                    hidden = Variable(hidden.data)
                optimizer.zero_grad()
                loss = loss_fn(prediction, y)
                loss.backward()
                optimizer.step()
                print("Epoch: %2d loss: %.6f lr: %f" % (epoch, loss, lr))

    print('\n', '#' * 10, 'Done', '#' * 10)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-a', '--architecture',
                           default=control_net,
                           help='Network Architecture to use')
    argparser.add_argument('--epochs',
                           default=5,
                           type=int,
                           help='number of total epochs to run')
    argparser.add_argument('-bs', '--batch-size',
                           default=128,
                           type=int,
                           dest='batch_size',
                           help='mini-batch size (default: 128)')
    argparser.add_argument('-lr', '--learning-rate',
                           default=0.0003,
                           type=float,
                           dest='lr',
                           help='initial learning rate')
    argparser.add_argument('--sanity-check',
                           dest='sanity_check',
                           action='store_true',
                           help='Sanity check')
    argparser.set_defaults(sanity_check=False)

    args = argparser.parse_args()

    control_net = init(args.architecture)
    try:
        if args.sanity_check:
            sanity_check(args.network, np.random.rand(10, 128, 256, 3), np.random.rand(10))
        else:
            train(args.network, np.random.rand(10, 128, 256, 3), np.random.rand(10))
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
