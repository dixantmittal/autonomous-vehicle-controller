import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torchvision.models import resnet


class ControlNet(Module):

    def __init__(self, model_path):
        super(ControlNet, self).__init__()

        resnet_base = list(resnet.resnet18(pretrained=True).children())[:-2]
        self.resnet_base = nn.Sequential(*resnet_base)

        # reduce resnet features to a vector of 2048 dimension (for this image size)
        self.reduce = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)

        self.bn_reduce = nn.BatchNorm2d(num_features=64)

        # output controls directly from this vector
        self.controls = nn.Linear(2048, 1)

        self.model_path = model_path

    def forward(self, inputs, hidden=None):
        inputs = inputs.permute(0, 3, 1, 2)
        n_batch, n_channel, n_height, n_width = inputs.size()

        features = F.dropout(F.leaky_relu(self.resnet_base(inputs)), 0.5)

        reduce = self.reduce(features)
        bn_reduce = F.dropout(F.leaky_relu(self.bn_reduce(reduce)), 0.5)

        flatten = bn_reduce.view(n_batch, -1)
        controls = self.controls(flatten)

        return controls, None


class RecurrentControlNet(Module):

    def __init__(self, model_path, hidden_size=2048):
        super(RecurrentControlNet, self).__init__()

        # extract input image features
        resnet18_conv = list(resnet.resnet18(pretrained=True).children())[:-2]
        self.resnet_base = nn.Sequential(*resnet18_conv)

        # reduce the features to a vector of 2048 dims
        self.reduce = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)

        # batch normalization for reduced form
        self.bn_reduce = nn.BatchNorm2d(num_features=64)

        self.features = nn.Linear(2048, hidden_size)

        self.bn_features = nn.BatchNorm1d(num_features=2048)

        # pass this 2048 vector to GRU along with hidden vector of dims 2048
        self.gru = nn.GRU(hidden_size, hidden_size, 1)

        # batch normalization for encoded output
        self.bn_gru = nn.BatchNorm1d(num_features=2048)

        # output the controls directly from GRU's output
        self.controls = nn.Linear(hidden_size, 1)

        self.model_path = model_path

    #
    # Inputs: input, h_0
    #       - **input** (batch, input_size): tensor containing input images. Batch is deemed to be sequential and is
    #                                            internally split into a sequential length of 10.
    #       - **h_0** (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
    #
    # Outputs: output, h_n
    #       - **output** (batch, 1): tensor containing the output steering control.
    #       - **h_n** (batch, hidden_size): tensor containing the hidden state for t=seq_len

    def forward(self, inputs, hidden_state=None):
        inputs = inputs.permute(0, 3, 1, 2)
        n_batch, n_channel, n_height, n_width = inputs.size()
        n_seq = min(10, n_batch)
        n_batch = n_batch // n_seq

        resnet = F.dropout(F.leaky_relu(self.resnet_base(inputs)), 0.5)

        reduce = self.reduce(resnet)
        bn_reduce = F.dropout(F.leaky_relu(self.bn_reduce(reduce)), 0.5)

        flatten = bn_reduce.view(n_seq * n_batch, -1)

        features = F.dropout(F.leaky_relu(self.bn_features(self.features(flatten))), 0.5)

        gru_out, hidden_state = self.gru(features, hidden_state)
        bn_gru = F.dropout(F.leaky_relu(self.bn_gru(gru_out.view(n_batch * n_seq, -1))), 0.5)

        controls = self.controls(bn_gru).view(n_seq, n_batch, -1)

        return controls, hidden_state
