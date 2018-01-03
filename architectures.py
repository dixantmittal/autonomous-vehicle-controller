import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torchvision.models import resnet


class ControlNet(Module):

    def __init__(self, model_path, image_dims=(3, 128, 256)):
        super(ControlNet, self).__init__()
        c, h, w = image_dims
        resnet18_conv = list(resnet.resnet18(pretrained=True).children())[:-2]
        self.resnet_base = nn.Sequential(*resnet18_conv)
        self.reduce = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        self.controls = nn.Linear(64 * h * w // 1024, 1)
        self.model_path = model_path

    def forward(self, inputs, hidden=None):
        inputs = inputs.permute(0, 3, 1, 2)
        n_batch, n_channel, n_height, n_width = inputs.size()
        features = F.dropout(F.leaky_relu(self.resnet_base(inputs)), 0.5)
        reduce = F.dropout(F.leaky_relu(self.reduce(features)), 0.5)
        flatten = reduce.view(n_batch, -1)
        controls = self.controls(flatten)

        return controls, None


class RecurrentControlNet(Module):

    def __init__(self, model_path, hidden_size=2048, image_dims=(3, 128, 256)):
        super(RecurrentControlNet, self).__init__()
        c, h, w = image_dims
        resnet18_conv = list(resnet.resnet18(pretrained=True).children())[:-2]
        self.resnet_base = nn.Sequential(*resnet18_conv)
        self.reduce = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        self.gru = nn.GRU(64 * h * w // 1024, hidden_size, 1, dropout=0.5)
        self.controls = nn.Linear(hidden_size, 1)
        self.model_path = model_path

    """
        Inputs: input, h_0
            - **input** (seq_len, batch, input_size): tensor containing the features
              of the input sequence. The input can also be a packed variable length
              sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
              for details.
            - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
              containing the initial hidden state for each element in the batch.

        Outputs: output, h_n
            - **output** (seq_len, batch, hidden_size * num_directions): tensor
              containing the output features h_t from the last layer of the RNN,
              for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
              given as the input, the output will also be a packed sequence.
            - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
              containing the hidden state for t=seq_len
    """

    def forward(self, inputs, hidden_state=None):
        inputs = inputs.permute(0, 3, 1, 2)
        n_batch, n_channel, n_height, n_width = inputs.size()
        n_seq = min(10, n_batch)
        n_batch = n_batch // n_seq

        resnet = F.dropout(F.leaky_relu(self.resnet_base(inputs)), 0.5)
        reduce = F.dropout(F.leaky_relu(self.reduce(resnet)), 0.5)
        flatten = reduce.view(n_seq, n_batch, -1)
        gru_out, hidden_state = self.gru(flatten, hidden_state)
        controls = self.controls(F.leaky_relu(gru_out.view(n_batch * n_seq, -1))).view(n_seq, n_batch, -1)
        return controls, hidden_state
