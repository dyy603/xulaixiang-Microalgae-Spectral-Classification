# mla.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, rope_dim):
        super().__init__()
        assert rope_dim % 4 == 0, f"dim ({rope_dim}) must be divisible by 4"
        self.rope_dim = rope_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dim // 4, dtype=torch.float32) / (rope_dim // 4)))
        self.register_buffer("inv_freq", inv_freq)

    def _compute_pe(self, h, w):
        pos_h = torch.arange(h, device=self.inv_freq.device)
        pos_w = torch.arange(w, device=self.inv_freq.device)

        sin_h = torch.sin(torch.einsum('i,j->ij', pos_h, self.inv_freq))
        cos_h = torch.cos(torch.einsum('i,j->ij', pos_h, self.inv_freq))

        sin_w = torch.sin(torch.einsum('i,j->ij', pos_w, self.inv_freq))
        cos_w = torch.cos(torch.einsum('i,j->ij', pos_w, self.inv_freq))

        pe = torch.zeros(h, w, self.rope_dim, device=self.inv_freq.device)
        pe[..., 0::4] = sin_h.unsqueeze(1).expand(-1, w, -1)
        pe[..., 1::4] = cos_h.unsqueeze(1).expand(-1, w, -1)
        pe[..., 2::4] = sin_w.unsqueeze(0).expand(h, -1, -1)
        pe[..., 3::4] = cos_w.unsqueeze(0).expand(h, -1, -1)
        return pe.unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        # assert C == self.dim, f"输入通道数{C}应与编码维度{self.dim}一致"
        pe = self._compute_pe(H, W)
        return pe.view(1, H * W, self.rope_dim // 2, 2)


def apply_rotary_emb(x, freqs_cis):
    x_ = x.unflatten(-1, (-1, 2))
    # freqs_cis = freqs_cis.view_as(x_)
    x_real = x_[..., 0]
    x_imag = x_[..., 1]
    freq_real = freqs_cis[..., 0]
    freq_imag = freqs_cis[..., 1]
    rotated_real = x_real * freq_real - x_imag * freq_imag
    rotated_imag = x_real * freq_imag + x_imag * freq_real
    rotated = torch.stack([rotated_real, rotated_imag], dim=-1)
    return rotated.flatten(-2, -1)


class ImageMLA(nn.Module):
    """图像版多头潜在注意力（适配二维特征图）"""

    def __init__(self, dim, num_heads=4, kv_rank=64, q_rank=32):
        super().__init__()
        assert dim % 4 == 0, f"Input channels {dim} must be divisible by 4"
        self.dim = dim
        self.num_heads = num_heads
        self.kv_rank = kv_rank
        self.q_rank = q_rank

        # 位置编码维度划分
        self.qk_rope_dim = dim // 2  # 25%维度用于位置编码
        self.qk_nope_dim = dim // 2  # - self.qk_rope_dim  # 非位置部分
        self.v_dim = dim // 2

        # 查询投影（LoRA分解）
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, q_rank, 1),
            nn.GroupNorm(1, q_rank),
            nn.Conv2d(q_rank, num_heads * (self.qk_nope_dim + self.qk_rope_dim), 1)
        )

        # 键值联合投影（共享低秩结构）
        self.kv_proj = nn.Sequential(
            nn.Conv2d(dim, kv_rank + self.qk_rope_dim, 1),
            nn.GroupNorm(1, kv_rank + self.qk_rope_dim),
        )
        self.kv_norm = nn.GroupNorm(1, kv_rank)
        self.kv_up = nn.Conv2d(kv_rank, num_heads * (self.qk_nope_dim + self.v_dim), 1)

        # 输出投影
        self.out_proj = nn.Conv2d(num_heads * self.v_dim, dim, 1)

        # 动态缩放因子
        self.scale = (self.qk_nope_dim + self.qk_rope_dim) ** -0.5
        self.rotary_pe = RotaryPositionalEncoding(self.qk_rope_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim, f"Input channels {C} != initialized dim {self.dim}"
        q_freqs = self.rotary_pe(x)
        q_freqs = q_freqs.expand(B, -1, -1, -1)  # [B, H*W, d, 2]
        q_freqs = q_freqs.unsqueeze(1)  # [B, 1, H*W, d, 2]
        q_freqs = q_freqs.expand(-1, self.num_heads, -1, -1, -1)  # [B, h, H*W, d, 2]

        # 查询投影
        q = self.q_proj(x)
        q = q.view(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        q_rope = apply_rotary_emb(q_rope, q_freqs)  # 传入调整后的q_freqs
        q = torch.cat([q_nope, q_rope], dim=-1)

        # 键值投影
        kv = self.kv_proj(x)  # [B, kv_rank+rope_dim, H, W]
        k_rope, kv = torch.split(kv, [self.qk_rope_dim, self.kv_rank], dim=1)
        kv = self.kv_norm(kv)
        kv = self.kv_up(kv)  # [B, h*(qk_nope+v_dim), H, W]
        kv = kv.view(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_dim, self.v_dim], dim=-1)

        # 合并键的位置编码
        k_rope = apply_rotary_emb(k_rope.view(B, self.qk_rope_dim, H * W)
                                  .permute(0, 2, 1)
                                  .unsqueeze(1), q_freqs)  # [B, 1, N, rope_dim]
        k = torch.cat([k_nope, k_rope.expand(-1, self.num_heads, -1, -1)], dim=-1)

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 值聚合
        v = (attn @ v).transpose(2, 3).contiguous().view(B, -1, H, W)
        return self.out_proj(v) + x  # 残差连接