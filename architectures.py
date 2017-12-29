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


class SimpleControlNet(Module):

    def __init__(self, model_path, image_dims=(3, 128, 256)):
        super(SimpleControlNet, self).__init__()
        c, h, w = image_dims
        # block 1
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.mp1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)

        # block 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.mp2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)

        # block 3
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.mp3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)

        # block 4
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.flatten = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1)

        self.fc1 = nn.Linear((h // 8 - 1) * (w // 8 - 1), 2048)
        self.controls = nn.Linear(2048, 1)
        self.model_path = model_path

    def forward(self, inputs, hidden=None):
        inputs = inputs.permute(0, 3, 1, 2)
        n_batch, n_channel, n_height, n_width = inputs.size()

        conv1 = F.leaky_relu(self.conv1(inputs))
        conv2 = F.leaky_relu(self.conv2(conv1))
        mp1 = F.leaky_relu(self.mp1(conv2))

        conv3 = F.leaky_relu(self.conv3(mp1))
        mp2 = F.leaky_relu(self.mp2(conv3))

        conv4 = F.leaky_relu(self.conv4(mp2))
        mp3 = F.leaky_relu(self.mp3(conv4))

        conv5 = F.leaky_relu(self.conv5(mp3))
        flatten = F.leaky_relu(self.flatten(conv5)).view(n_batch, -1)

        fc1 = F.dropout(F.leaky_relu(self.fc1(flatten)), 0.5)
        controls = self.controls(fc1)

        return controls, None


class RecurrentControlNet(Module):

    def __init__(self, model_path, hidden_size=2048, image_dims=(3, 128, 256)):
        super(RecurrentControlNet, self).__init__()
        c, h, w = image_dims
        resnet18_conv = list(resnet.resnet18(pretrained=True).children())[:-1]
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
        n_seq = 10
        inputs = inputs.permute(0, 3, 1, 2)
        n_batch, n_channel, n_height, n_width = inputs.size()
        n_batch = n_batch // n_seq

        resnet = F.dropout(F.leaky_relu(self.resnet_base(inputs)), 0.5)
        reduce = F.dropout(F.leaky_relu(self.reduce(resnet)), 0.5)
        flatten = reduce.view(n_seq, n_batch, -1)
        gru_out, hidden_state = self.gru(flatten, hidden_state)
        controls = self.controls(F.leaky_relu(gru_out.view(n_batch * n_seq, -1))).view(n_seq, n_batch, -1)
        return controls, hidden_state
