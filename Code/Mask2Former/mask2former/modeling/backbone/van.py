import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torch.hub import load_state_dict_from_url

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m): 
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim, neuron_factor, multilayer_perceptron=False):
        super().__init__()
        self.conv0 = nn.Conv2d(dim # In channels
                               , dim # Out channels
                               , 5 # Kernel size
                               , padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        self.multilayer_perceptron = multilayer_perceptron


        if multilayer_perceptron:
            self.mlp = ChannelMLP(dim, neuron_factor)
        else:
            self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        if self.multilayer_perceptron:
            attn = self.mlp(attn)
        else:
            attn = self.conv1(attn)

        return u * attn


class ChannelMLP(nn.Module):
    def __init__(self, dim, neuron_factor):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * neuron_factor),
            nn.GELU(),
            nn.Linear(dim * neuron_factor, dim * neuron_factor),
            nn.GELU(),
            nn.Linear(dim * neuron_factor, dim)
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, a=-0.5, b=0.5)  # Uniform in range [-0.5, 0.5]
            nn.init.zeros_(layer.bias)  # Bias set to 0

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)                # (B, H, W, C)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)                # Back to (B, C, H, W)
        return x

class MultiBranchLKA(nn.Module):
    def __init__(self, dim, neuron_factor, multilayer_perceptron=False):
        super().__init__()
        
        

        # Branch 1: original LKA
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.multilayer_perceptron = multilayer_perceptron
        if multilayer_perceptron:
            self.mlp = ChannelMLP(dim, neuron_factor)
        else:
            self.conv1 = nn.Conv2d(dim, dim, 1)


        # Branch 2: different dilation and kernel
        self.branch2 = nn.Sequential( # K=12 d=2
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 6, stride=1, padding=5, groups=dim, dilation=2),
            nn.Conv2d(dim, dim, 1) if not multilayer_perceptron else ChannelMLP(dim, neuron_factor)
        )

        # Branch 3: even more spread
        self.branch3 = nn.Sequential( # K=15 d=3
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3),
            nn.Conv2d(dim, dim, 1) if not multilayer_perceptron else ChannelMLP(dim, neuron_factor)
        )

        # Learnable weights (softmaxed during forward)
        self.weights = nn.Parameter(torch.ones(3))

        self.start_temp = 5
        self.end_temp = 1.0
        iterations = 20000 # This is the number of iterations to reach the end temperature.
        self.total_steps = iterations # That is the number of steps to reach the end temperature.
        self.step_counter = torch.tensor(0, dtype=torch.float32)  # Initialize step counter

    def compute_temperature(self):
        progress = min(self.step_counter.item() / self.total_steps, 1.0)
        temp = self.end_temp + (self.start_temp - self.end_temp) * math.exp(-self.start_temp * progress)
        return temp

    def forward(self, x):
        u = x.clone()

        out1 = self.conv0(x)
        out1 = self.conv_spatial(out1)
        if self.multilayer_perceptron:
            out1 = self.mlp(out1)
        else:
            out1 = self.conv1(out1)
        out2 = self.branch2(x)
        out3 = self.branch3(x)


        temperature = self.compute_temperature()
        # Normalize weights
        norm_weights = torch.softmax(self.weights / temperature, dim=0)
        
        # Update step counter
        self.step_counter += 1

        # Weighted sum
        attn = norm_weights[0] * out1 + norm_weights[1] * out2 + norm_weights[2] * out3
        
        return u * attn  # same shape as input


class Attention(nn.Module):
    def __init__(self, d_model, neuron_factor,multibranch=False, multilayer_perceptron=False):
        super().__init__()
        
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        if multibranch:
            self.spatial_gating_unit = MultiBranchLKA(d_model, neuron_factor, multilayer_perceptron)
        else:
            self.spatial_gating_unit = LKA(d_model, neuron_factor, multilayer_perceptron)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class BatchGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=4, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        N, C, H, W = x.shape
        D = C * H * W
        G = self.num_groups
        S = D // G

        x_flat = x.view(N, -1)           # (N, D)
        x_grouped = x_flat.view(N, G, S) # (N, G, S)

        mean = x_grouped.mean(dim=(0, 2), keepdim=True) # (1, G, 1)
        var = x_grouped.var(dim=(0, 2), keepdim=True, unbiased=False) # (1, G, 1)

        x_norm = (x_grouped - mean) / (var + self.eps).sqrt() # (N, G, S)
        x_norm = x_norm.view(N, D).view(N, C, H, W) # (N, C, H, W)

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm


class Block(nn.Module): # Here is a full one step of VAN.
    def __init__(self, dim, neuron_factor, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, multibranch=False, multilayer_perceptron=False, normalization="BatchNorm"):
        super().__init__()

        if normalization == "BatchNorm":
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        elif normalization == "GroupNorm":
            self.norm1 = nn.GroupNorm(dim//8, dim)
            self.norm2 = nn.GroupNorm(dim//8, dim)
        elif normalization == "BatchGroupNorm":
            self.norm1 = BatchGroupNorm(dim, num_groups=dim//8) # For small batch sizes, the research paper suggests using num_groups=1. If it doesnt work I will try something else. I cant use G=1 for memory reasons.
            self.norm2 = BatchGroupNorm(dim, num_groups=dim//8)

        self.attn = Attention(dim, neuron_factor,multibranch, multilayer_perceptron)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module): # Its applied before each stage of VAN. Its used to create the patch embeddings.
    """ Image to Patch Embedding
    """

    def __init__(self, stage, normalization, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        
        if stage == 1 or normalization == "BatchNorm":
            self.norm = nn.BatchNorm2d(embed_dim)
        elif normalization == "GroupNorm":
            self.norm = nn.GroupNorm(embed_dim//8, embed_dim)
        elif normalization == "BatchGroupNorm":
            self.norm = BatchGroupNorm(embed_dim, num_groups=embed_dim//8)
            
        

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

class VAN(nn.Module):
    def __init__(self, neuron_factor = 4,img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False, pretrained=None, multibranch=False, multilayer_perceptron=False, normalization="BatchNorm"):
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6) if norm_layer == "LayerNormEps1e-6" else norm_layer
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], stage=i+1, normalization=normalization)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], neuron_factor = neuron_factor,mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j],multibranch=multibranch, multilayer_perceptron=multilayer_perceptron, normalization = normalization)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        

        self.apply(self._init_weights)

        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, pretrained):
        if isinstance(pretrained, str):
            pretrained = load_state_dict_from_url(pretrained, map_location="cpu")
            state_dict = pretrained["state_dict"]
            if self.num_classes == 0:
                state_dict.pop("head.weight", None)
                state_dict.pop("head.bias", None)
                strict = False
            else:
                strict = True

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict)
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("\033[92mPretrained weights for VAN backbone loaded successfully.\033[0m")

        else:
            print("\033[91mWarning: Pretrained weights for VAN backbone loaded with missing or unexpected keys.\033[0m")
            print("Missing keys: ", missing_keys)
            print("Unexpected keys: ", unexpected_keys)

        
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        feature_maps = {}

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            feature_maps[f"res{i + 2}"] = x
            

        return feature_maps

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

@BACKBONE_REGISTRY.register()
class D2VAN(VAN, Backbone):
    def __init__(self, cfg, input_shape): # Detectron2 register function expects 2 a cfg and an input shape argument. Thats why we cant only put cfg.

        embed_dims = cfg.MODEL.VAN.EMBED_DIMS
        mlp_ratios = cfg.MODEL.VAN.MLP_RATIOS
        depths = cfg.MODEL.VAN.DEPTHS
        num_stages = cfg.MODEL.VAN.NUM_STAGES
        img_size = cfg.MODEL.VAN.IMG_SIZE
        in_chans = cfg.MODEL.VAN.IN_CHANS
        flag = cfg.MODEL.VAN.FLAG
        num_classes = cfg.MODEL.VAN.NUM_CLASSES
        norm_layer = cfg.MODEL.VAN.NORM_LAYER
        drop_rate = cfg.MODEL.VAN.DROP_RATE
        drop_path_rate = cfg.MODEL.VAN.DROP_PATH_RATE
        pretrained_van = cfg.MODEL.VAN.PRETRAINED_VAN
        multibranch = cfg.MODEL.VAN.MULTIBRANCH_LKA
        multilayer_perceptron = cfg.MODEL.VAN.MULTILAYER_PERCEPTRON
        normalization = cfg.MODEL.VAN.NORMALIZATION
        neuron_factor = cfg.MODEL.VAN.NEURON_FACTOR

        super().__init__(
            neuron_factor=neuron_factor,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dims=embed_dims,
            mlp_ratios=mlp_ratios,
            depths=depths,
            num_stages=num_stages,
            flag=flag,
            norm_layer=norm_layer,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained_van,
            multibranch=multibranch,
            multilayer_perceptron=multilayer_perceptron,
            normalization=normalization
        )

        self._out_features = cfg.MODEL.VAN.OUT_FEATURES


        self._out_feature_strides = { # How much the spatial resolution is downsampled at each feature stage.
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = { # The number of channels in each feature stage.
            "res2": embed_dims[0],
            "res3": embed_dims[1],
            "res4": embed_dims[2],
            "res5": embed_dims[3],
        }





    def forward(self, x):
            """
            Args:
                x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
            Returns:
                dict[str->Tensor]: names and the corresponding features
            """
            assert (
                x.dim() == 4
                ),  f"VAN takes an input of shape (N, C, H, W). Got {x.shape} instead!" # N is the batch size, C is the number of channels, H is the height, and W is the width.
            outputs = {}
            y = super().forward(x)
            for k in y.keys():
                if k in self._out_features:
                    outputs[k] = y[k]
            return outputs
        
    def output_shape(self):
            return {
                name: ShapeSpec(
                    channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
                )
                for name in self._out_features
            }

    @property
    def size_divisibility(self):
        return 32