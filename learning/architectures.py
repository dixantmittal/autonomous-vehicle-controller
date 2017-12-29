import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torchvision.models import resnet


class ControlNet(Module):

    def __init__(self, model_path):
        super(ControlNet, self).__init__()
        resnet18_conv = list(resnet.resnet18(pretrained=True).children())[:-1]
        self.resnet_base = nn.Sequential(*resnet18_conv)
        self.fc1 = nn.Linear(9216, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.controls = nn.Linear(512, 3)
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

    def forward(self, inputs, hidden=None):
        inputs = inputs.permute(0, 3, 1, 2)
        n_batch, n_channel, n_height, n_width = inputs.size()
        features = self.resnet_base(inputs)
        flatten = features.view(n_batch, -1)
        fc1 = F.dropout(F.relu(self.fc1(flatten)), 0.5)
        fc2 = F.dropout(F.relu(self.fc2(fc1)), 0.5)
        controls = self.controls(fc2)
        return controls, None


class RecurrentControlNet(Module):

    def __init__(self, model_path, hidden_size=1024):
        super(RecurrentControlNet, self).__init__()
        resnet18_conv = list(resnet.resnet18(pretrained=True).children())[:-1]
        self.resnet_base = nn.Sequential(*resnet18_conv)
        self.gru = nn.GRU(9216, hidden_size, 1)
        self.controls = nn.Linear(hidden_size, 3)
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
        if len(inputs.size()) < 5:
            n_seq = 10
            n_batch, n_height, n_width, n_channel = inputs.size()
            inputs = inputs.view(-1, n_seq, n_height, n_width, n_channel)
        inputs = inputs.permute(0, 1, 4, 2, 3)
        n_batch, n_seq, n_channel, n_height, n_width = inputs.size()
        features = self.resnet_base(inputs.contiguous().view(-1, n_channel, n_height, n_width))
        flatten = features.view(n_batch * n_seq, -1)
        gru_out, hidden_state = self.gru(flatten.view(n_seq, n_batch, -1), hidden_state)
        controls = self.controls(F.relu(gru_out.view(n_batch * n_seq, -1))).view(n_seq, n_batch, -1)
        return controls, hidden_state
