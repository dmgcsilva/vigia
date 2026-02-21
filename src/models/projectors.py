from torch import nn
import torch
from torchvision.ops import MLP

# connector class to connect the vision encoder to the language decoder
# this is a generic class that is then extended by specific connector classes
class Connector(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Connector is an abstract class and cannot be used directly.")



# Simple Linear layer connector, equal to the one used in the MM-PlanLLM paper
class LinearConnector(Connector):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# MLP connector of varying depth, similar to recent Llava work
class MLPConnector(Connector):
    def __init__(self, in_dim:int, out_dim:int, connector_depth:int=2, dropout:float=0.0, layer_norm:bool=True, **kwargs):
        super().__init__()
        connector_hidden = 2 ** (int((in_dim + out_dim) / 2).bit_length() - 1)
        # create the MLP
        self.model = MLP(
            in_channels=in_dim,
            hidden_channels=[connector_hidden]*(connector_depth-1) + [out_dim], # the last layer should have the same size as the LLM embeddings
            norm_layer=nn.LayerNorm if layer_norm else None,
            activation_layer=nn.GELU,
            bias=True,
            dropout=dropout
        )

    def forward(self, x):
        return self.model(x)


# Connector class that uses a transformer block to connect the vision encoder to the language decoder
class TransformerConnector(Connector):
    def __init__(self, in_dim, out_dim, connector_depth=2, dropout=0.0, **kwargs):
        super().__init__()

        # connector hidden is the half way point between the input and output dimensions (but still a power of 2)
        connector_hidden = 2 ** (int((in_dim + out_dim) / 2).bit_length() - 1)
        # create the transformer block
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=8,
                dim_feedforward=connector_hidden,
                dropout=dropout,
                activation='gelu'
            ),
            num_layers=connector_depth
        )

        # output layer to match the LLM embeddings
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.out(self.model(x))
    

CONNECTOR_REGISTRY = {
    'linear': LinearConnector,
    'mlp': MLPConnector,
    'transformer': TransformerConnector
}



class RetrievalProjector(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        raise NotImplementedError("RetrievalProjector is an abstract class and cannot be used directly.")


class SingleRetrievalProjector(RetrievalProjector):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if kwargs.get('dropout', 0.0) > 0.0:
            self.dropout = nn.Dropout(kwargs['dropout'])
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        if self.dropout is not None:
            y = self.dropout(y)
        return y, None
    

# Simple Linear layer retrieval projector, equal to the one used in the MM-PlanLLM paper
class LinearRetrievalProjector(RetrievalProjector):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.start_fc = nn.Linear(in_dim, out_dim)
        self.end_fc = nn.Linear(in_dim, out_dim)

        if kwargs.get('dropout', 0.0) > 0.0:
            self.start_dropout = nn.Dropout(kwargs['dropout'])
            self.end_dropout = nn.Dropout(kwargs['dropout'])
        else:
            self.start_dropout = None
            self.end_dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_start = self.start_fc(x)
        y_end = self.end_fc(x)

        if self.start_dropout is not None:
            y_start = self.start_dropout(y_start)
            y_end = self.end_dropout(y_end)

        return y_start, y_end


# MLP retrieval projector of fixed depth, similar to recent Llava work
# TODO: see if a funnel like MLP would work better, i.e. decreasing hidden size instead of the current setup which is IN -> IN -> OUT
#   this can lead to too aggressive dimensionality reduction (4096 -> 4096 -> 512), which might be too much  
class MLPRetrievalProjector(RetrievalProjector):
    def __init__(self, in_dim, out_dim, dropout=0.0, layer_norm=True, **kwargs):
        super().__init__()
        mid_hidden = 2 ** (int((in_dim + out_dim) / 2).bit_length() - 1)
        # create the MLP
        self.start_model = MLP(
            in_channels=in_dim,
            hidden_channels=[mid_hidden, out_dim], # just two layers
            norm_layer=nn.LayerNorm if layer_norm else None,
            activation_layer=nn.GELU,
            bias=True,
            dropout=dropout
        )

        self.end_model = MLP(
            in_channels=in_dim,
            hidden_channels=[mid_hidden, out_dim], # just two layers
            norm_layer=nn.LayerNorm if layer_norm else None,
            activation_layer=nn.GELU,
            bias=True,
            dropout=dropout
        )

    def forward(self, x):
        return self.start_model(x), self.end_model(x)


class MLPSplitRetrievalProjector(RetrievalProjector):
    def __init__(self, in_dim, out_dim, dropout=0.0, layer_norm=True, **kwargs):
        super().__init__()

        hidden_dim = kwargs.get('hidden_dim', min(in_dim, out_dim*2))
        mid_hidden = 2 ** (int((in_dim + hidden_dim) / 2).bit_length() - 1)

        # create the MLP
        self.model = MLP(
            in_channels=in_dim,
            hidden_channels=[mid_hidden, hidden_dim], # just two layers
            norm_layer=nn.LayerNorm if layer_norm else None,
            activation_layer=nn.GELU,
            bias=True,
            dropout=dropout
        )

        self.start_fc = nn.Linear(hidden_dim, out_dim)
        self.end_fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        y = self.model(x)
        return self.start_fc(y), self.end_fc(y)
    

RETRIEVAL_PROJECTOR_REGISTRY = {
    'single': SingleRetrievalProjector,
    'linear': LinearRetrievalProjector,
    'mlp': MLPRetrievalProjector,
    'mlp_split': MLPSplitRetrievalProjector
}