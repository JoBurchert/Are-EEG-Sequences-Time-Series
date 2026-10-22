import torch
import torch.nn as nn
from vit_pytorch import ViT
from vit_pytorch.vit import pair, Rearrange, Transformer

class ViTWrapper(ViT):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        # Hard copy from the original constructor, but without the patch size assert
        # We crop the series in the forward pass

        super(ViT, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_size = patch_size

        #assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: This only works without subject embedding at the moment
        if x.shape[-1] % self.patch_size[1] != 0:
            crop = self.patch_size[1] * (x.shape[-1] // self.patch_size[1])
            x = x[:, :, :crop]

        if len(x.shape) < 4:
            # add artificial image heigth
            x = x.unsqueeze(1)
        return super(ViTWrapper, self).forward(x)



def get_vit(dataset, in_channels, num_pred_classes, patch_size=25, dropout=None, depth=None, mlp_dim=None)->nn.Module:
    if dataset.lower() == 'mamem':
        timeseries_length = 125
        dropout = dropout or 0.1
        depth = depth or 2
        mlp_dim = mlp_dim or 256
    elif dataset.lower() == 'bci':
        timeseries_length = 438
        raise NotImplementedError
    elif dataset.lower() == 'bcicha':
        timeseries_length = 160
        dropout = dropout or 0.0
        depth = depth or 4
        mlp_dim = mlp_dim or 512
    else:
        raise NotImplementedError()
    model = ViTWrapper(
        image_size=(in_channels, timeseries_length),
        patch_size=(1, patch_size), # patches with height 1
        channels=1,
        num_classes=num_pred_classes,
        dim=1, # output is a sequence of length 1
        depth=depth,
        heads=8,
        mlp_dim=mlp_dim,
        dropout=dropout,
        #emb_dropout=0.1
    )
    # testing the model
    # x = torch.ones(size=(1, in_channels, 1, dataloader.timeseries_length))
    # y = model(x)
    return model
