import torch
import torch.nn as nn
class SpectralFrontEnd(nn.Module):
    def __init__(self, sig_len, use_filter=True, n_filters=4):
        super().__init__()
        self.use_filter = bool(use_filter)
        F = n_filters if self.use_filter else 1
        self.F = F
        if self.use_filter:
            w = torch.randn(F, sig_len, 2) * 0.02
            w[..., 0] += 1.0
            self.weight = nn.Parameter(w)
        self.out_ch = 2 * F

    def forward(self, x):
        x = x.float()
        xc = torch.complex(x[:, 0], x[:, 1])
        p = xc.abs().pow(2).mean(-1, keepdim=True).clamp_min(1e-12)
        xc = xc / p.sqrt()
        Xf = torch.fft.fftshift(torch.fft.fft(xc, dim=-1), dim=-1)
        if self.use_filter:
            w = torch.view_as_complex(self.weight.contiguous())
            Y = Xf.unsqueeze(1) * w.unsqueeze(0)
        else:
            Y = Xf.unsqueeze(1)
        return torch.cat([Y.real, Y.imag], dim=1)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.h = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.drop(self.proj(out))


class FormerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================ SpectralFormer ============================
class SpectralFormer(nn.Module):
    def __init__(self, num_classes, sig_len, dim=128, depth=2, heads=4,
                 patch=64, dropout=0.3, use_filter=True, n_filters=4):
        super().__init__()
        assert sig_len % patch == 0, "sig_len must be divisible by patch"
        N = sig_len // patch
        self.fe = SpectralFrontEnd(
            sig_len, use_filter=use_filter, n_filters=n_filters)
        self.embed = nn.Conv1d(self.fe.out_ch, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, N, dim))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([FormerBlock(dim, heads, 4, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        x = self.embed(self.fe(x)).transpose(1, 2)
        x = self.drop(x + self.pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


def build_model(args, num_classes):
    return SpectralFormer(
        num_classes=num_classes, sig_len=args.sig_len,
        dim=args.dim, depth=args.depth, heads=args.heads,
        patch=args.patch, dropout=args.dropout,
        use_filter=bool(getattr(args, "use_filter", 1)),
        n_filters=getattr(args, "n_filters", 4),
    )
