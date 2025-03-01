import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
import math
from backbone.Global_reranking_module import Global_ranktuning
from backbone.salad import SALAD


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.work_with_tokens = work_with_tokens

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1. / p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class Flatten(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """

    def __init__(self, args):
        super(GeoLocalizationNet, self).__init__()
        self.backbone = get_backbone(args)
        self.salad = SALAD(num_clusters=args.num_cluster,token_dim=args.features_dim*2,cluster_dim=args.features_dim)
        self.Global_ranktuning = Global_ranktuning(train_batch_size=4, N=4, sort_num_heads=4, encoder_layer=2, sort_layers=4, num_corr=15,
                 sort_dim=64, num_class=2)

    def forward(self, x):
        _, _, H, W = x.shape
        H = H // 14
        W = W // 14
        x = self.backbone(x)
        B, _, D = x["x_prenorm"].shape
        x0 = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)
        x = (x_p, x0)
        global_features = self.salad(x)
        return global_features


def get_backbone(args):
    backbone = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)
    assert not (args.foundation_model_path is None and args.resume is None), "Please specify foundation model path."
    if args.foundation_model_path:
        model_dict = backbone.state_dict()
        state_dict = torch.load(args.foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone


if __name__ == "__main__":
    from parser import parse_arguments

    args = parse_arguments()
    input = torch.rand(4, 3, 224, 224)
    model = GeoLocalizationNet(args=args)
    output = model(input)
    print(output.shape)
