import torch
from torch import nn

from modules.TDNNLayer 				 import TDNN as TDNNLayer
from modules.StatisticsPoolingLayers import StatisticsPooling

class TDNNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, node_num, context, full_context, dropout=0.4,
                 attentive=True, verbose=True, device="cuda:0", **kwargs):

        """
        TDNN Neural network model with attentive statistics pooling layer
        """

        # Batch normalization (avoid corruption after backward)
        # Dropout (to avoid overfiting)

        super(TDNNetwork,self).__init__()

        # Input and output size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Activation functions
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)

        ######## FRAME LEVEL TDNN LAYERS ##########
        self.tdnn1 = nn.Sequential(TDNNLayer(context[0], input_dim, node_num[0], full_context[0]),
                                   nn.BatchNorm1d(node_num[0]),
                                   self.activation,
                                   self.dropout)

        self.tdnn2 = nn.Sequential(TDNNLayer(context[1], node_num[0], node_num[1], full_context[1]),
                                   nn.BatchNorm1d(node_num[1]),
                                   self.activation,
                                   self.dropout)

        self.tdnn3 = nn.Sequential(TDNNLayer(context[2], node_num[1], node_num[2], full_context[2]),
                                   nn.BatchNorm1d(node_num[2]),
                                   self.activation,
                                   self.dropout)

        self.tdnn4 = nn.Sequential(TDNNLayer(context[3], node_num[2], node_num[3], full_context[3]),
                                   nn.BatchNorm1d(node_num[3]),
                                   self.activation,
                                   self.dropout)

        self.tdnn5 = nn.Sequential(TDNNLayer(context[4], node_num[3], node_num[4], full_context[4]),
                                   nn.BatchNorm1d(node_num[4]),
                                   self.activation,
                                   self.dropout)

        ######## STATISTICS POOLING LAYER ##########
        self.stats_pooling_layer = StatisticsPooling(node_num[4],1,attentive)

        ####### UTTERANCE LEVEL LAYERS #########
        self.x_vector = nn.Linear(node_num[5], node_num[6])

        self.x_vector_layers = nn.Sequential(nn.BatchNorm1d(node_num[6]),
                                             self.activation,
                                             self.dropout)

        self.linear = nn.Sequential(nn.Linear(node_num[6], node_num[7]),
                                    nn.BatchNorm1d(node_num[7]),
                                    self.activation,
                                    self.dropout)

        self.output_layer = nn.Sequential(nn.Linear(node_num[7], output_dim))

        # ADDITIONAL INFO
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if verbose:
            print("This model will use ",self.device)

    def forward(self, x, extraction=False):

        """
        Forward pass
        :param x:
        :return x_utt:
        """

        # Get dims
        batch_size, seq_len, _ = x.size()

        # Pre-process
        x = x.view(-1, self.input_dim)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, -1)

        # TDNN Layers
        x = self.tdnn1(x).transpose(1,2).contiguous() # TDNN 1
        x = self.tdnn2(x).transpose(1,2).contiguous() # TDNN 2
        x = self.tdnn3(x).transpose(1,2).contiguous() # TDNN 3
        x = self.tdnn4(x).transpose(1,2).contiguous() # TDNN 4
        x = self.tdnn5(x).transpose(1,2).contiguous() # TDNN 5

        # Attentive statistics pooling layer
        x = self.stats_pooling_layer(x)

        # LINEAR LAYERS
        x = self.x_vector(x) # Extract X-vector here

        if extraction:
            return x

        x = self.x_vector_layers(x)
        x = self.linear(x)  # LINEAR 2

        # Output Layer
        return self.output_layer(x)

