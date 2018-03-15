import numpy as np
import torch


def get_data(dir, start, end):
    X = None
    y = None
    for b in range(start, end):

        # if b % 3 == 0:
        #     continue
        # train for data batches
        try:
            if X is None:
                X = np.load(dir + '/input_image_{:0>5d}.npy'.format(b))
                y = np.load(dir + '/controls_{:0>5d}.npy'.format(b))
            else:
                X = np.append(X, np.load(dir + '/input_image_{:0>5d}.npy'.format(b)), axis=0)
                y = np.append(y, np.load(dir + '/controls_{:0>5d}.npy'.format(b)), axis=0)
        except FileNotFoundError:
            print('input_image_{:0>5d}.npy'.format(b) + ' Not Found!')
            continue

        print('input_image_{:0>5d}.npy'.format(b) + ' Loaded!')

    if X is not None:
        print('Input Data Shape: ', X.shape)
        print('Input Controls Shape: ', y.shape)
        print()

    return X, y


def save(model):
    print('>> Saving Model... << ')
    print('> Model Path: ', model.model_path)
    torch.save({
        'state_dict': model.state_dict()
    }, model.model_path)
    print('> Model Saved Successfully!!')


def normalize_image(X=None):
    if X is not None:
        X = X / 128 - 1

    return X
