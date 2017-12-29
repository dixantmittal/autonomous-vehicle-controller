import _pickle
import argparse

import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from architectures import ControlNet, RecurrentControlNet

recurrent_control_net = 'recurrent_control_net'
control_net = 'control_net'

networks = {
    recurrent_control_net: RecurrentControlNet('model/recurrentcontrolnet_rgb_steering_throttle.pt'),
    control_net: ControlNet('model/controlnet_rgb_steering_throttle.pt')
}


class CustomDataSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        pass

    def __getitem__(self, index):
        X_batch = self.X[index]
        y_batch = self.y[index]

        return X_batch, y_batch

    def __len__(self):
        return len(self.X)


def get_data(start, end):
    X = None
    y = None
    for b in range(start, end):

        # train for data batches
        if X is None:
            X = np.load(args.dir + '/input_image_{:0>5d}.npy'.format(b))
            y = np.load(args.dir + '/controls_{:0>5d}.npy'.format(b))
        else:
            X = np.append(X, np.load(args.dir + '/input_image_{:0>5d}.npy'.format(b)), axis=0)
            y = np.append(y, np.load(args.dir + '/controls_{:0>5d}.npy'.format(b)), axis=0)

        print('input_image_{:0>5d}.npy'.format(b) + ' Loaded!')
    print('\nInput Data Shape: ', X.shape)
    print('Input Controls Shape: ', y.shape)

    return X, y


def normalize_data(X=None, y=None):
    # if X is not None:
    #     X = X / 128 - 1

    if y is not None:
        # normalize for speed at 20kph
        y[range(y.shape[0]), 2] = y[range(y.shape[0]), 2] / 10 - 1
        # normalize throttle
        y[range(y.shape[0]), 1] = y[range(y.shape[0]), 1] * 2 - 1

    return X, y


def unnormalize_data(X=None, y=None):
    # if X is not None:
    #     X = (X + 1) * 128

    if y is not None:
        # normalize for speed at 20kph
        y[range(y.shape[0]), 2] = (y[range(y.shape[0]), 2] + 1) * 10
        # normalize throttle
        y[range(y.shape[0]), 1] = (y[range(y.shape[0]), 1] + 1) / 2

    return X, y


def flip_data(X=None, y=None):
    if X is not None:
        dims = X.shape
        X = np.flip(X, axis=dims[-2])

    if y is not None:
        y[range(y.shape[0]), 0] = -y[range(y.shape[0]), 0]

    return X, y


def train(network, batch_start, batch_end):
    distributed_model = model = networks[network]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    cudnn.benchmark = True

    try:
        print('\n====> Model Path: ', model.model_path)
        model_state = torch.load(model.model_path)
        model.load_state_dict(model_state['state_dict'])
        optimizer.load_state_dict(model_state['optimizer'])
        print('====> Model found and loaded!!')
    except (OSError, _pickle.UnpicklingError):
        print('\n====> Model not found. Training from scratch...')

    if torch.cuda.is_available():
        distributed_model = model = model.cuda()

        if args.multi_gpu:
            print('====> Multiple GPUs mode: [ON]')
            distributed_model = torch.nn.DataParallel(model)

        loss_fn = loss_fn.cuda()

    print()
    print('#' * 5, 'Starting Training', '#' * 5)

    for epoch in range(args.epochs):
        print('\n==> Starting epoch: ', epoch + 1)

        start = batch_start
        while start < batch_end:
            if batch_end - start > 15:
                end = start + 15
            else:
                end = batch_end

            X, y = get_data(start, end)
            X, y = normalize_data(X, y)

            # start training
            dataloader = DataLoader(dataset=CustomDataSet(X, y),
                                    batch_size=args.batch_size,
                                    num_workers=10)
            loss = 0
            hidden = None
            for itr, batch in tqdm(enumerate(dataloader)):
                X, y = Variable(batch[0], requires_grad=False).cuda(), Variable(batch[1], requires_grad=False).cuda()

                prediction, hidden = distributed_model(X, hidden)
                if hidden is not None:
                    hidden = Variable(hidden.data)

                optimizer.zero_grad()
                loss = loss_fn(prediction, y)

                loss.backward()

                optimizer.step()

            print('==> loss: ', loss.data[0])
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model.model_path)
            start = end
    print()
    print('#' * 5, 'Training Finished', '#' * 5)
    print()


def predict_controls(network, X):
    model = networks[network]
    X = Variable(torch.from_numpy(X), requires_grd=False)
    if torch.cuda.is_available():
        model = model.cuda()
        X = X.cuda()
    return model(X)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--network', default='control_net',
                           help='architecture to use (default: control_net)')
    argparser.add_argument('--epochs', default=5, type=int, metavar='N',
                           help='number of total epochs to run')
    argparser.add_argument('--batch-size', default=128, type=int, metavar='N', dest='batch_size',
                           help='mini-batch size (default: 256)')
    argparser.add_argument('--learning-rate', default=0.1, type=float, metavar='LR', dest='lr',
                           help='initial learning rate')
    argparser.add_argument('--start-set', type=int, default=0, dest='start_set',
                           help='starting image set to load (default: 0)')
    argparser.add_argument('--end-set', type=int, default=299, dest='end_set',
                           help='ending image set to load (default: 299)')
    argparser.add_argument('--set-dir', default='data_rgb', dest='dir',
                           help='directory for image set to load (default: data_rgb)')
    argparser.add_argument('--multi-gpu', dest='multi_gpu', action='store_true',
                           help='Use multiple gpus')
    argparser.set_defaults(multi_gpu=False)

    args = argparser.parse_args()

    train(args.network, args.start_set, args.end_set + 1)
