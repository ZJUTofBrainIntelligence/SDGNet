import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utils: log densities
# =========================
def log_normal_diag(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """log N(x | mean, diag(exp(logvar))) per sample. x, mean, logvar: [B,D] -> [B]"""
    return -0.5 * (math.log(2 * math.pi) + logvar + (x - mean) ** 2 / torch.exp(logvar)).sum(dim=-1)


def log_standard_normal(x: torch.Tensor) -> torch.Tensor:
    """log N(x | 0, I) per sample. x: [B,D] -> [B]"""
    return -0.5 * (math.log(2 * math.pi) + x ** 2).sum(dim=-1)


# =========================
# Cross-attention fusion
# =========================
class CrossAttnFusion(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d, d)
        )
        self.ln2 = nn.LayerNorm(d)

    def forward(self, T, S):
        # T as Query, S as Key/Value
        attn_out, _ = self.attn(query=T, key=S, value=S, need_weights=False)
        H = self.ln(T + attn_out)
        H2 = self.ln2(H + self.ffn(H))
        return H2


# =========================
# Lightweight ST Encoder
# =========================
class STEncoderLite(nn.Module):
    """
    Input:  x [B,L,C]
    Output: mu/logvar [B,D]
    """
    def __init__(
        self,
        L: int = 256,
        C: int = 8,
        d: int = 128,
        D: int = 128,
        k_t: int = 5,
        k_c: int = 3,
        dropout: float = 0.1,
        use_3rd_layer: bool = False,
    ):
        super().__init__()
        self.L, self.C, self.d, self.D = L, C, d, D

        # Temporal branch
        t_layers = [
            nn.Conv1d(C, d, kernel_size=k_t, padding=k_t // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d, d, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if use_3rd_layer:
            t_layers += [nn.Dropout(dropout), nn.Conv1d(d, d, kernel_size=3, padding=1), nn.GELU()]
        self.t_conv = nn.Sequential(*t_layers)
        self.t_norm = nn.LayerNorm(d)

        # Spatial branch (per time-step along channel axis)
        s_layers = [
            nn.Conv1d(1, d, kernel_size=k_c, padding=k_c // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d, d, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if use_3rd_layer:
            s_layers += [nn.Dropout(dropout), nn.Conv1d(d, d, kernel_size=3, padding=1), nn.GELU()]
        self.s_conv = nn.Sequential(*s_layers)
        self.s_norm = nn.LayerNorm(d)

        self.fusion = CrossAttnFusion(d=d, n_heads=4, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)  # over time
        self.fc_mu = nn.Linear(d, D)
        self.fc_logvar = nn.Linear(d, D)

    def forward(self, x: torch.Tensor):
        B, L, C = x.shape
        assert L == self.L and C == self.C

        # Temporal: [B,L,C] -> [B,C,L] -> conv -> [B,d,L] -> [B,L,d]
        T = self.t_conv(x.transpose(1, 2)).transpose(1, 2)
        T = self.t_norm(T)

        # Spatial: [B,L,C] -> [B*L,1,C] -> conv -> [B*L,d,C] -> mean_C -> [B,L,d]
        S = x.reshape(B * L, 1, C)
        S = self.s_conv(S)
        S = S.mean(dim=-1).reshape(B, L, self.d)
        S = self.s_norm(S)

        # Cross-attn fusion
        H = self.fusion(T, S)  # [B,L,d]

        # Pool over time -> [B,d]
        h_pool = self.pool(H.transpose(1, 2)).squeeze(-1)

        mu = self.fc_mu(h_pool)
        logvar = torch.clamp(self.fc_logvar(h_pool), -10.0, 10.0)
        return mu, logvar


# =========================
# Synergy-Decomposed Decoder
# =========================
class TempPrototypeDec(nn.Module):
    """z [B,D] -> g [B,L,r]"""
    def __init__(self, L: int = 256, D: int = 128, r: int = 8, L0: int = 32, h0: int = 128):
        super().__init__()
        assert L % L0 == 0
        self.L, self.D, self.r, self.L0, self.h0 = L, D, r, L0, h0

        self.fc = nn.Linear(D, L0 * h0)

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv1d(h0, h0 // 2, 3, padding=1), nn.GELU())
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv1d(h0 // 2, h0 // 4, 3, padding=1), nn.GELU())
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv1d(h0 // 4, h0 // 8, 3, padding=1), nn.GELU())
        self.out = nn.Conv1d(h0 // 8, r, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.fc(z).view(B, self.L0, self.h0).transpose(1, 2)  # [B,h0,L0]
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)                            # [B,h0/8,L]
        g = self.out(h).transpose(1, 2)            # [B,L,r]
        return g


class SynergyDec(nn.Module):
    """z [B,D] -> M [B,r,C]"""
    def __init__(self, D: int = 128, r: int = 8, C: int = 8, hidden: int = 128):
        super().__init__()
        self.r, self.C = r, C
        self.mlp = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Linear(hidden, r * C),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        return self.mlp(z).view(B, self.r, self.C)


class SynergyDecoder(nn.Module):
    """z -> g (Lxr), z -> M (rxC), xhat = g @ M -> [B,L,C]"""
    def __init__(self, L: int = 256, C: int = 8, D: int = 128, r: int = 8, refine: bool = True):
        super().__init__()
        self.tempdec = TempPrototypeDec(L=L, D=D, r=r, L0=32, h0=128)
        self.sydec = SynergyDec(D=D, r=r, C=C, hidden=128)

        self.use_refine = refine
        if refine:
            self.refine = nn.Sequential(
                nn.Conv1d(C, C, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(C, C, kernel_size=3, padding=1),
            )

    def forward(self, z: torch.Tensor):
        g = self.tempdec(z)            # [B,L,r]
        M = self.sydec(z)              # [B,r,C]
        xhat = torch.matmul(g, M)      # [B,L,C]
        if self.use_refine:
            xhat = xhat + self.refine(xhat.transpose(1, 2)).transpose(1, 2)
        return xhat, g, M


# =========================
# Shared: compute log_w & normalized weights
# =========================
def compute_logw_and_wnorm(
    x: torch.Tensor,          # [B,L,C]
    xhats: torch.Tensor,      # [K,B,L,C]
    z_samples: torch.Tensor,  # [K,B,D]
    mu: torch.Tensor,         # [B,D]
    logvar: torch.Tensor,     # [B,D]
    sigma_x: float = 1.0,
):
    """
    log_w_k = log p(x|z_k) + log p(z_k) - log q(z_k|x)
    Then w_norm = softmax(log_w) over k.
    (IWAE core definition) :contentReference[oaicite:2]{index=2}
    """
    K, B, L, C = xhats.shape

    # log p(x|z): isotropic Gaussian
    recon_sq = (x.unsqueeze(0) - xhats).pow(2).sum(dim=(2, 3))  # [K,B]
    log_px_z = -0.5 * (recon_sq / (sigma_x ** 2)) - 0.5 * (L * C) * math.log(2 * math.pi * (sigma_x ** 2))

    # log p(z)
    log_pz = torch.stack([log_standard_normal(z_samples[k]) for k in range(K)], dim=0)  # [K,B]

    # log q(z|x)
    log_qz_x = torch.stack([log_normal_diag(z_samples[k], mu, logvar) for k in range(K)], dim=0)  # [K,B]

    log_w = log_px_z + log_pz - log_qz_x  # [K,B]
    w_norm = torch.softmax(log_w, dim=0)  # [K,B]
    return log_w, w_norm


# =========================
# IWAE Loss (+ optional ortho reg)
# =========================
def iwae_loss(
    log_w: torch.Tensor,   # [K,B]
    Ms: list,              # list length K, each [B,r,C]
    ortho_lambda: float = 1e-3,
) -> torch.Tensor:
    K, B = log_w.shape
    # IWAE bound: E[ log( 1/K * sum_k w_k ) ] = logsumexp(log_w) - log K :contentReference[oaicite:3]{index=3}
    log_mean_w = torch.logsumexp(log_w, dim=0) - math.log(K)  # [B]
    loss = -(log_mean_w.mean())

    # Ortho reg on M: ||M^T M - I||_F^2
    if ortho_lambda > 0.0:
        reg = 0.0
        for M in Ms:  # [B,r,C]
            MtM = torch.matmul(M.transpose(1, 2), M)  # [B,C,C]
            I = torch.eye(MtM.size(-1), device=MtM.device, dtype=MtM.dtype).unsqueeze(0)
            reg = reg + (MtM - I).pow(2).mean()
        reg = reg / len(Ms)
        loss = loss + ortho_lambda * reg

    return loss


# =========================
# ST-IWAE Model
# =========================
class STIWAETwitter(nn.Module):
    def __init__(
        self,
        L: int = 256,
        C: int = 8,
        d: int = 128,
        D: int = 128,
        r: int = 8,
        K: int = 5,
        dropout: float = 0.1,
        use_3rd_layer: bool = False,
        refine: bool = True,
    ):
        super().__init__()
        self.L, self.C, self.d, self.D, self.r, self.K = L, C, d, D, r, K
        self.encoder = STEncoderLite(L=L, C=C, d=d, D=D, dropout=dropout, use_3rd_layer=use_3rd_layer)
        self.decoder = SynergyDecoder(L=L, C=C, D=D, r=r, refine=refine)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, K: int):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(K, mu.size(0), mu.size(1), device=mu.device, dtype=mu.dtype)
        z = mu.unsqueeze(0) + std.unsqueeze(0) * eps
        return z, eps

    def forward(self, x: torch.Tensor, sigma_x: float = 1.0):
        """
        Returns (10 items, consistent with __main__ unpack):
          xhat_iw, xhat_mean, xhats, w_norm, log_w, mu, logvar, z_samples, gs, Ms
        """
        mu, logvar = self.encoder(x)                             # [B,D]
        z_samples, _ = self.reparameterize(mu, logvar, self.K)   # [K,B,D]

        xhats, gs, Ms = [], [], []
        for k in range(self.K):
            xhat_k, g_k, M_k = self.decoder(z_samples[k])
            xhats.append(xhat_k)
            gs.append(g_k)
            Ms.append(M_k)

        xhats = torch.stack(xhats, dim=0)  # [K,B,L,C]
        xhat_mean = xhats.mean(dim=0)      # [B,L,C]

        log_w, w_norm = compute_logw_and_wnorm(
            x=x, xhats=xhats, z_samples=z_samples, mu=mu, logvar=logvar, sigma_x=sigma_x
        )
        # importance-weighted reconstruction: sum_k softmax(log_w)_k * xhat_k
        xhat_iw = (w_norm.unsqueeze(-1).unsqueeze(-1) * xhats).sum(dim=0)  # [B,L,C]

        return xhat_iw, xhat_mean, xhats, w_norm, log_w, mu, logvar, z_samples, gs, Ms

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        return self.encoder(x)


# =========================
# Quick sanity check
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, C = 16, 256, 8
    D, K, r = 128, 5, 8

    model = STIWAETwitter(L=L, C=C, d=128, D=D, r=r, K=K, use_3rd_layer=False, refine=True)
    x = torch.randn(B, L, C)

    # forward
    xhat_iw, xhat_mean, xhats, w_norm, log_w, mu, logvar, z_samples, gs, Ms = model(x, sigma_x=1.0)

    print("input:", x.shape)
    print("xhat_iw:", xhat_iw.shape)
    print("xhat_mean:", xhat_mean.shape)
    print("xhats:", xhats.shape)
    print("w_norm:", w_norm.shape)
    print("log_w:", log_w.shape)
    print("z_samples:", z_samples.shape)
    print("mu/logvar:", mu.shape, logvar.shape)

    loss = iwae_loss(log_w, Ms, ortho_lambda=1e-3)
    print("loss:", float(loss))
