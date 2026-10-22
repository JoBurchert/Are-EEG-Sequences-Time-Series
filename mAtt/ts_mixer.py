import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtsmixer.layers import MixerLayer, TimeBatchNorm2d, feature_to_time, time_to_feature


class TSMixer(nn.Module):
    """
    All credits to the torchTSMixer package
    """

    def __init__(
        self,
        dataset: str,
        input_channels: int,
        output_channels: int = None,
        activation_fn: str = "relu",
        prediction_length=1,
        num_blocks=None,
        dropout_rate=None,
        ff_dim=None,
        normalize_before: bool = True,
        norm_type: str = "batch",
    ):
        super().__init__()

        if dataset.lower() == 'mamem':
            sequence_length = 125
            dropout_rate = dropout_rate or 0.1
            num_blocks = num_blocks or 4
            ff_dim = ff_dim or 256
        elif dataset.lower() == 'bci':
            sequence_length = 438
            dropout_rate = dropout_rate or 0.0
            num_blocks = num_blocks or 4
            ff_dim = ff_dim or 128
        elif dataset.lower() == 'bcicha':
            sequence_length = 160
            dropout_rate = dropout_rate or 0.1
            num_blocks = num_blocks or 4
            ff_dim = ff_dim or 256
        else:
            raise NotImplementedError()

        # Transform activation_fn to callable
        activation_fn = getattr(F, activation_fn)

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # Build mixer layers
        self.mixer_layers = self._build_mixer(
            num_blocks,
            input_channels,
            output_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            sequence_length=sequence_length,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

        # Temporal projection layer
        self.temporal_projection = nn.Linear(sequence_length, prediction_length)


    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        x_hist = x_hist.permute(0, 2, 1)
        x = self.mixer_layers(x_hist)

        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)

        return x.squeeze(1)


    def _build_mixer(
        self, num_blocks: int, input_channels: int, output_channels: int, **kwargs
    ):
        """Build the mixer blocks for the model.

        Args:
            num_blocks (int): Number of mixer blocks to be built.
            input_channels (int): Number of input channels for the first block.
            output_channels (int): Number of output channels for the last block.
            **kwargs: Additional keyword arguments for mixer layer configuration.

        Returns:
            nn.Sequential: Sequential container of mixer layers.
        """
        output_channels = output_channels if output_channels is not None else input_channels
        channels = [input_channels] * (num_blocks - 1) + [output_channels]

        return nn.Sequential(
            *[
                MixerLayer(input_channels=in_ch, output_channels=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )


class TSMixerNoSub(TSMixer):
    def forward(self, x: torch.Tensor, x_sub: torch.Tensor):
        return super().forward(x)


class TSMixerHead(TSMixer):
    def __init__(
        self,
        dataset: str,
        input_channels: int,
        output_channels: int = None,
        activation_fn: str = "relu",
        prediction_length=1,
        num_blocks=None,
        dropout_rate=None,
        ff_dim=None,
        normalize_before: bool = True,
        norm_type: str = "batch",
    ):
        super().__init__()
        raise NotImplementedError

        if dataset.lower() == 'mamem':
            sequence_length = 125
            dropout_rate = dropout_rate or 0.1
            num_blocks = num_blocks or 4
            ff_dim = ff_dim or 256
        elif dataset.lower() == 'bci':
            sequence_length = 438
            dropout_rate = dropout_rate or 0.0
            num_blocks = num_blocks or 4
            ff_dim = ff_dim or 128
        elif dataset.lower() == 'bcicha':
            sequence_length = 160
            dropout_rate = dropout_rate or 0.1
            num_blocks = num_blocks or 4
            ff_dim = ff_dim or 256
        else:
            raise NotImplementedError()

        # Transform activation_fn to callable
        activation_fn = getattr(F, activation_fn)

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # Build mixer layers
        self.mixer_layers = self._build_mixer(
            num_blocks,
            input_channels,
            output_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            sequence_length=sequence_length,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

        # Temporal projection layer
        self.temporal_projection = nn.Linear(sequence_length, prediction_length)


    def forward(self, x_hist: torch.Tensor, x_sub: torch.Tensor) -> torch.Tensor:
        x_hist = x_hist.permute(0, 2, 1)
        x = self.mixer_layers(x_hist)

        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)

        return x.squeeze(1)


class TSMixerEmb(TSMixer):
    def __init__(
        self,
        dataset: str,
        input_channels: int,
        output_channels: int,
        num_subjects:int,
        activation_fn: str = "relu",
        prediction_length=1,
        num_blocks=None,
        dropout_rate=None,
        ff_dim=None,
        normalize_before: bool = True,
        norm_type: str = "batch",
    ):
        self.num_subjects = num_subjects
        input_channels += num_subjects
        super().__init__(dataset, input_channels, output_channels, activation_fn,
                         prediction_length, num_blocks, dropout_rate, ff_dim, normalize_before, norm_type)


    def forward(self, x_hist: torch.Tensor, x_sub: torch.Tensor) -> torch.Tensor:
        x_hist = torch.cat([x_hist, x_sub], dim=1)
        x_hist = x_hist.permute(0, 2, 1)
        x = self.mixer_layers(x_hist)

        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)

        return x.squeeze(1)


class TSMixerIdEmb(TSMixer):
    def __init__(
        self,
        dataset: str,
        input_channels: int,
        output_channels: int,
        num_subjects:int,
        activation_fn: str = "relu",
        prediction_length=1,
        num_blocks=None,
        dropout_rate=None,
        ff_dim=None,
        normalize_before: bool = True,
        norm_type: str = "batch",
    ):
        self.num_subjects = num_subjects
        emb_dim = 16
        input_channels += emb_dim
        super().__init__(dataset, input_channels, output_channels, activation_fn,
                         prediction_length, num_blocks, dropout_rate, ff_dim, normalize_before, norm_type)
        self.subject_embeddings = nn.Embedding(num_subjects + 1, emb_dim, padding_idx=0)


    def forward(self, x_hist: torch.Tensor, x_sub: torch.Tensor) -> torch.Tensor:
        z_sub = self.subject_embeddings(x_sub.int())
        z_sub = torch.squeeze(z_sub, dim=1).permute(0, 2, 1)
        x_hist = torch.cat([x_hist, z_sub], dim=1)
        x_hist = x_hist.permute(0, 2, 1)
        x = self.mixer_layers(x_hist)

        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)

        return x.squeeze(1)
