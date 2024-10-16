import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from ..encoders import twins_svt_large, convnext_large
from .twins import PosConv
from .encoder import MemoryEncoder
from .decoder import MemoryDecoder
from .cnn import BasicEncoder, ResidualBlock
from .SAM_encoder import get_encoder, get_encoder_base, get_encoder_tiny
from .CAM import LTSE, ContextAdapter, WeightedAdding

class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        
        H1, W1, H2, W2 = cfg.pic_size
        H_offset = (H1-H2) // 2
        W_offset = (W1-W2) // 2
        cfg.H_offset = H_offset
        cfg.W_offset = W_offset

        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain, del_layers=cfg.del_layers)
        elif "jh" in cfg.cnet:
            self.context_encoder = twins_svt_large_jihao(pretrained=self.cfg.pretrain, del_layers=cfg.del_layers, version=cfg.cnet)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        elif cfg.cnet == 'convnext':
            self.context_encoder = convnext_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'nat':
            self.context_encoder = nat_base(pretrained=self.cfg.pretrain)

        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
        if cfg.sam_scale == 'H':
            self.sam_encoder = get_encoder(ft_ckpt=cfg.ft_ver, checkpoint=cfg.sam_checkpoint)
        elif cfg.sam_scale == 'B':
            self.sam_encoder = get_encoder_base(checkpoint=cfg.sam_checkpoint)
        else:
            self.sam_encoder = get_encoder_tiny(checkpoint=cfg.sam_checkpoint)
        self.sam_encoder.requires_grad_(False)

        if cfg.weighted_add:
            self.wadd_1 = WeightedAdding()
        
        self.up_layer8 = nn.Sequential(
            nn.Conv2d(256,96,3,1,1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        
        self.CFM = nn.Sequential(ResidualBlock(256+96, 256), ResidualBlock(256, 256))

        self.LTSE = LTSE(256)
        self.CAM = ContextAdapter(256)

        if self.cfg.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.requires_grad_(False)
                    m.eval()

        if cfg.pretrain_mode:
            print("[In pretrain mode, freeze context encoder]")
            for param in self.context_encoder.parameters():
                param.requires_grad = False


    def train(self, mode):
        super().train(mode)
        self.sam_encoder.eval()
        if self.cfg.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.eval()

    def forward(self, image1, image2, mask=None, output=None, flow_init=None):
        if self.cfg.pretrain_mode:
            loss = self.pretrain_forward(image1, image2, mask=mask, output=output)
            return loss
        else:
            # SAM encoder
            with torch.no_grad():
                sam_feat = self.sam_encoder((image1 - self.pixel_mean) / self.pixel_std)
            
            sam_feat = self.up_layer8(sam_feat)

            # Following https://github.com/princeton-vl/RAFT/
            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0

            data = {}
            
            context, _ = self.context_encoder(image1)
            context_quater = None

            # CFM
            context = self.CFM(torch.cat([context, sam_feat], dim=1))
            
            # LTSE
            b,_,h,w = image1.shape
            sparse_embedding = self.LTSE((h,w), b)

            # CAM
            context_add = self.CAM(context, sparse_embedding)

            if self.cfg.weighted_add:
                context = self.wadd_1(context, context_add)
            else:
                context = context + context_add

            cost_memory, cost_patches, feat_s_quater, feat_t_quater = self.memory_encoder(image1, image2, data, context)

            flow_predictions = self.memory_decoder(cost_memory, context, context_quater, feat_s_quater, feat_t_quater, data, flow_init=flow_init, cost_patches=cost_patches)

            return flow_predictions
    
    def pretrain_forward(self, image1, image2, mask=None, output=None, flow_init=None):
        # SAM encoder
        with torch.no_grad():
            sam_feat = self.sam_encoder((image1 - self.pixel_mean) / self.pixel_std)
        
        sam_feat = self.up_layer8(sam_feat)

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        H_offset = self.cfg.H_offset
        W_offset = self.cfg.W_offset
        H2, W2 = self.cfg.pic_size[2:]
        
        image1_inner = image1[:, :, H_offset:H_offset+H2, W_offset:W_offset+W2]
        image2_inner = image2[:, :, H_offset:H_offset+H2, W_offset:W_offset+W2]
        
        data = {}
        
        context, _ = self.context_encoder(image1_inner)
            
        # CFM
        context = self.CFM(torch.cat([context, sam_feat], dim=1))
        
        # LTSE
        b,_,h,w = image1.shape
        sparse_embedding = self.LTSE((h,w), b)

        # CAM
        context_add = self.CAM(context, sparse_embedding)

        if self.cfg.weighted_add:
            context = self.wadd_1(context, context_add)
        else:
            context = context + context_add

        cost_memory, cost_patches = self.memory_encoder.pretrain_forward(image1, image2, image1_inner, image2_inner, data, context , mask=mask)

        loss = self.memory_decoder.pretrain_forward(cost_memory, context, data, flow_init=flow_init, cost_patches=cost_patches)

        return loss
