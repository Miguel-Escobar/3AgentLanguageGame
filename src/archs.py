# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from egg.core.interaction import Interaction
from egg.core.transformer import TransformerEncoder, TransformerDecoder


class Agent:
    def __init__(
        self,
        embedding_dim: int,
        max_len: int,
        numSpeakerLayers: int,
        numSpeakerHeads: int,
        hiddenSizeSpeaker: int,
    ):
        self.Listener = TransformerDecoder(
            embed_dim=embedding_dim,
            max_len=max_len,
            num_layers=numListenerLayers,
            num_heads=numListenerHeads,
            hidden_size=hiddenSizeListener,
        )
        self.Speaker = TransformerEncoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embedding_dim,
            num_heads=numSpeakerHeads,
            hidden_size=hiddenSizeSpeaker,
            num_layers=numSpeakerLayers,
            positional_embedding=pos_embedding,
        )
        self.Vision = ...
        self.Action = ...

        self.Sender = ...
        self.Receiver = ...


class Speaker(nn.Module):
    def __init__(self):
        super(Speaker, self).__init__()


class Listener(nn.Module):
    pass


class Vision(nn.Module):
    def __init__(self):
        super(Vision, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return x


class Action(nn.Module):
    def __init__(self):
        super(Action, self).__init__()
        pass


class PretrainNet(nn.Module):
    def __init__(self, vision_module):
        super(PretrainNet, self).__init__()
        self.vision_module = vision_module
        self.fc = nn.Linear(500, 10)

    def forward(self, x):
        x = self.vision_module(x)
        x = self.fc(F.leaky_relu(x))
        return x


class Sender(nn.Module):
    def __init__(self):
        super(Sender, self).__init__()

    def forward(
        self,
        sender_input: torch.Tensor,
        aux_input: Dict[str, torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        pass


class Receiver(nn.Module):
    def __init__(self):
        super(Receiver, self).__init__()

    def forward(
        self,
        message: torch.Tensor,
        receiver_input: torch.Tensor = None,
        aux_input: Dict[str, torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        pass


class Game(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, Dict[str, Any]],
        ],
    ):
        super(Game, self).__init__()

    def forward(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        receiver_input: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Interaction]:
        pass
