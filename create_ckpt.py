import torch
import torch.nn as nn
import sys
sys.path.append('core')
from core.FlowFormer import build_flowformer
from configs.things import get_cfg

cfg = get_cfg()
model = nn.DataParallel(build_flowformer(cfg))
state_dict = model.state_dict()

# Load ours checkpoint
# change the path here
ours_state_dict = torch.load('logs/PATH-TO-FINAL-FILE/final', map_location='cpu')


for k in state_dict.keys():
    if k in ours_state_dict:
        if k[7:18] != 'sam_encoder' and k[7:16] != 'up_layer8' and k[7:10] != 'CFM' and k[7:11] != 'LTSE' and k[7:10] != 'CAM':
            state_dict[k] = ours_state_dict[k].clone()

model.load_state_dict(state_dict, strict=True)

torch.save(model.state_dict(), 'init-checkpoint.pth')