import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.query_lin = nn.Linear(feat_dim, feat_dim)
        self.key_lin   = nn.Linear(feat_dim, feat_dim)
        self.value_lin = nn.Linear(feat_dim, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, t, v):
        # t: (batch, feat_dim), v: (batch, num_regions, feat_dim)
        q = self.query_lin(t).unsqueeze(1)      # (batch, 1, feat_dim)
        k = self.key_lin(v)                     # (batch, num_regions, feat_dim)
        v_val = self.value_lin(v)               # (batch, num_regions, feat_dim)
        attn = torch.softmax(q @ k.transpose(-2,-1) / self.feat_dim**0.5, dim=-1)
        attended = attn @ v_val                 # (batch, 1, feat_dim)
        attended = attended.squeeze(1)          # (batch, feat_dim)
        fused = torch.cat([t, attended], dim=-1)  # (batch, 2*feat_dim)
        return fused
    
class MFB(nn.Module):
    def __init__(self, in_dim, out_dim, factor=5):
        super().__init__()
        self.factor = factor
        self.out_dim = out_dim
        self.lin_text  = nn.Linear(in_dim, out_dim * factor)
        self.lin_image = nn.Linear(in_dim, out_dim * factor)

    def forward(self, t, v):
        # t, v: (batch, in_dim)
        t_proj = self.lin_text(t)   # (batch, out_dim * factor)
        v_proj = self.lin_image(v)  # (batch, out_dim * factor)
        eltwise = t_proj * v_proj   # (batch, out_dim * factor)
        # reshape and sum-pool across factor dimension
        batch = eltwise.size(0)
        eltwise = eltwise.view(batch, self.out_dim, self.factor)
        fused = eltwise.sum(2)      # (batch, out_dim)
        # signed-sqrt + l2 norm (optional but recommended)
        fused = torch.sign(fused) * torch.sqrt(torch.abs(fused) + 1e-10)
        fused = F.normalize(fused, dim=-1)
        return fused
    

class GMU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lin_t    = nn.Linear(in_dim, hidden_dim)
        self.lin_v    = nn.Linear(in_dim, hidden_dim)
        self.lin_gate = nn.Linear(in_dim * 2, hidden_dim)

    def forward(self, t, v):
        # t, v: (batch, in_dim)
        t_h = torch.tanh(self.lin_t(t))
        v_h = torch.tanh(self.lin_v(v))
        gate_input = torch.cat([t, v], dim=-1)
        z = torch.sigmoid(self.lin_gate(gate_input))  # (batch, hidden_dim)
        fused = z * t_h + (1 - z) * v_h
        return fused
    

class ConcatFuser(nn.Module):
    """
    A simple modality fuser that just concatenates two feature tensors.
    """
    def __init__(self):
        super().__init__()


    def forward(self, text_embeds, image_embeds):
        """
        Args:
            text_embeds: Tensor of shape (batch_size, ..., feat_dim)
            image_embeds: Tensor of same shape as text_embeds
        Returns:
            fused: Tensor of shape (batch_size, ..., 2*feat_dim)
        """
        return torch.cat((text_embeds, image_embeds), dim=-1)


fuser_map = {
    'concat': lambda cfg: ConcatFuser(),
    'mfb':    lambda cfg: MFB(cfg.projection_dim, cfg.projection_dim, factor=getattr(cfg, 'factor', 5)),
    'gmu':    lambda cfg: GMU(cfg.projection_dim, cfg.projection_dim),
    'cross_attn': lambda cfg: CrossModalAttention(cfg.projection_dim),
}