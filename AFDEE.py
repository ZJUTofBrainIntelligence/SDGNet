import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# =================================================================================
# 1. UTILITY LAYERS & FUNCTIONS
# =================================================================================

class DropPath(nn.Module):
    """Implements DropPath (Stochastic Depth) regularization."""

    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    """
    将1D多通道时间序列转换为嵌入补丁序列。
    nn.Conv1d 自然支持多通道输入 (in_chans > 1)。
    """
    def __init__(self, seq_len: int, patch_size: int = 16, in_chans: int = 1, embed_dim: int = 256):
        super(PatchEmbed, self).__init__()
        # 使用一个合理的步长，例如 patch_size // 4
        stride = patch_size // 4
        self.num_patches = (seq_len - patch_size) // stride + 1
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状 [B, L, C_in]
        Returns:
            torch.Tensor: 输出张量，形状 [B, num_patches, embed_dim]
        """
        x = x.transpose(1, 2)  # Conv1d 需要 [B, C_in, L]
        x_proj = self.proj(x)
        x_out = x_proj.flatten(2).transpose(1, 2)
        return x_out


# =================================================================================
# 2. CORE TSLANET BUILDING BLOCKS
# =================================================================================

class Adaptive_Spectral_Block(nn.Module):
    """在频域中执行自适应滤波 / 带通滤波 / 简单能量阈值。"""

    def __init__(
        self,
        dim: int,
        enhancement_factor: float = 1.5,
        mode: str = "afdee",
        bp_low: float = 0.1,
        bp_high: float = 0.4,
        energy_ratio: float = 1.0,
    ):
        """
        mode:
            - "afdee"   : 原始自适应频域增强（论文方法）
            - "bandpass": 固定带通滤波
            - "energy"  : 简单能量阈值法
        """
        super(Adaptive_Spectral_Block, self).__init__()
        assert mode in ["afdee", "bandpass", "energy"], f"Unsupported spectral mode: {mode}"
        self.mode = mode
        self.bp_low = bp_low
        self.bp_high = bp_high
        self.energy_ratio = energy_ratio

        self.enhancement_factor = enhancement_factor
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        trunc_normal_(self.complex_weight_high, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))
        self.scaling_factor = nn.Parameter(torch.ones(1) * self.enhancement_factor)

    def create_adaptive_high_freq_mask(self, x_fft: torch.Tensor) -> torch.Tensor:
        B, N_fft, C = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        median_energy = energy.median(dim=1, keepdim=True)[0]
        normalized_energy = energy / (median_energy + 1e-6)
        sigmoid_scale = 10
        adaptive_mask = torch.sigmoid(sigmoid_scale * (normalized_energy - self.threshold_param))
        adaptive_mask = 1 + self.scaling_factor * adaptive_mask
        return adaptive_mask.unsqueeze(-1)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        x_in: [B, N, C]
        """
        B, N, C = x_in.shape
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, N_fft, C]
        N_fft = x_fft.shape[1]

        if self.mode == "bandpass":
            # --- 固定带通滤波：对频率轴使用固定窗口 ---
            freqs = torch.linspace(0, 1, steps=N_fft, device=x_fft.device, dtype=x_fft.real.dtype)
            bp_mask = ((freqs >= self.bp_low) & (freqs <= self.bp_high)).float()
            bp_mask = bp_mask.view(1, N_fft, 1)  # [1, N_fft, 1]
            x_fft_out = x_fft * bp_mask

        elif self.mode == "energy":
            # --- 简单能量阈值法：能量低于阈值的频率直接置零 ---
            # 先在通道维做求和得到每个频点的总能量 [B, N_fft]
            energy = torch.abs(x_fft).pow(2).sum(dim=-1)
            # 阈值 = energy_mean * ratio
            threshold = self.energy_ratio * energy.mean(dim=1, keepdim=True)
            mask = (energy >= threshold).float().unsqueeze(-1)  # [B, N_fft, 1]
            x_fft_out = x_fft * mask

        else:
            # --- 原始 AFDEE 自适应频域增强 ---
            weight_complex = torch.view_as_complex(self.complex_weight)  # [C]
            x_weighted = x_fft * weight_complex.view(1, 1, -1)  # [B, N_fft, C]

            mask = self.create_adaptive_high_freq_mask(x_fft)  # [B, N_fft, 1]
            x_enhanced = x_fft * mask  # [B, N_fft, C]

            weight_high_complex = torch.view_as_complex(self.complex_weight_high)  # [C]
            x_weighted_high = x_enhanced * weight_high_complex.view(1, 1, -1)

            x_fft_out = x_weighted + x_weighted_high

        x_out = torch.fft.irfft(x_fft_out, n=N, dim=1, norm='ortho')
        return x_out.to(x_in.dtype)


class ICB(nn.Module):
    """带门控的反向卷积块。"""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.):
        super(ICB, self).__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        x1 = self.act(self.conv1(x_t))
        x1 = self.drop(x1)
        x2 = self.act(self.conv2(x1))
        x2 = self.drop(x2)
        gated_out = x1 * x2
        x_out = self.conv3(gated_out)
        return x_out.transpose(1, 2)


class TSLANet_layer(nn.Module):
    """TSLANet模型的单层。"""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 1.,
        drop: float = 0.15,
        drop_path: float = 0.,
        spectral_mode: str = "afdee",
    ):
        super(TSLANet_layer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.asb = Adaptive_Spectral_Block(dim, mode=spectral_mode)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.asb(self.norm1(x)))
        x = x + self.drop_path(self.icb(self.norm2(x)))
        return x


# =================================================================================
# 3. 统一的多通道TSLANet模型 (MODIFIED WITH INTERPOLATION)
# =================================================================================

class UnifiedTSLANet(nn.Module):
    def __init__(
        self,
        in_seq_len: int,
        out_seq_len: int,
        in_chans: int,
        out_chans: int,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_blocks: int = 2,
        drop_rate: float = 0.15,
        use_spectral: bool = True,      # 预留，将来可以细化
        use_icb: bool = True,          # 预留
        use_pos_embed: bool = True,
        use_backbone: bool = True,
        spectral_mode: str = "afdee",
    ):
        super(UnifiedTSLANet, self).__init__()
        self.out_seq_len = out_seq_len

        # 1. 特征提取器 (TSLANet Backbone)
        self.feature_extractor = TSLANet(
            seq_len=in_seq_len,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            drop_rate=drop_rate,
            use_pos_embed=use_pos_embed,
            use_backbone=use_backbone,
            spectral_mode=spectral_mode,
        )

        # 2. 投影头 (Projection Head)
        self.head = nn.Linear(embed_dim, out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状 [B, In_Seq_Len, C_in]

        Returns:
            torch.Tensor: 输出张量，形状 [B, Out_Seq_Len, C_out]
        """
        # 1. 通过TSLANet提取特征
        # 输入: [B, 400, C_in] -> 输出: [B, num_patches, embed_dim]
        features = self.feature_extractor(x)

        # 2. 应用投影头，将嵌入维度映射到输出通道数
        # 输入: [B, num_patches, embed_dim] -> 输出: [B, num_patches, C_out]
        projected_features = self.head(features)

        # 3. 使用线性插值调整序列长度
        # a. 转置以匹配interpolate的输入格式 [B, C, L]
        # 输入: [B, num_patches, C_out] -> 输出: [B, C_out, num_patches]
        x_for_resampling = projected_features.transpose(1, 2)

        # b. 执行插值
        # 输入: [B, C_out, num_patches] -> 输出: [B, C_out, Out_Seq_Len]
        x_resampled = F.interpolate(x_for_resampling, size=self.out_seq_len, mode='linear', align_corners=False)

        # c. 转置回最终的目标格式 [B, L, C]
        # 输入: [B, C_out, Out_Seq_Len] -> 输出: [B, Out_Seq_Len, C_out]
        output = x_resampled.transpose(1, 2)

        return output


class TSLANet(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        num_blocks: int,
        drop_rate: float,
        use_pos_embed: bool = True,
        use_backbone: bool = True,
        spectral_mode: str = "afdee",
    ):
        super(TSLANet, self).__init__()

        self.use_pos_embed = use_pos_embed
        self.use_backbone = use_backbone

        self.patch_embed = PatchEmbed(
            seq_len=seq_len, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_rate, num_blocks)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(
                dim=embed_dim,
                mlp_ratio=1.,
                drop=drop_rate,
                drop_path=dpr[i],
                spectral_mode=spectral_mode,
            )
            for i in range(num_blocks)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.use_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if self.use_backbone:
            for blk in self.tsla_blocks:
                x = blk(x)

        return x


# =================================================================================
# 4. 示例用法 (UPDATED FOR FIXED INPUT AND OUTPUT SIZES)
# =================================================================================
if __name__ == "__main__":
    # --- 模型超参数 ---
    BATCH_SIZE = 16
    SEQ_LEN = 400          # 输入时间序列的长度 (固定)
    OUTPUT_SEQ_LEN = 256   # 目标输出时间序列的长度 (固定)
    IN_CHANNELS = 8        # 输入时间序列的通道数
    OUT_CHANNELS = 8       # 最终输出序列的通道数
    PATCH_SIZE = 16        # Patch大小
    EMBED_DIM = 256        # 内部嵌入维度

    # --- 实例化统一模型 ---
    model = UnifiedTSLANet(
        in_seq_len=SEQ_LEN,
        out_seq_len=OUTPUT_SEQ_LEN,
        in_chans=IN_CHANNELS,
        out_chans=OUT_CHANNELS,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
    )

    print("--- 统一模型架构 ---")
    print(model)

    # --- 计算并打印模型参数量 ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- 模型参数量 ---")
    print(f"模型总参数量: {total_params / 1e6:.2f} M")

    # --- 模拟前向传播 ---
    print("\n--- 模拟前向传播 ---")

    # 创建一个固定形状的模拟多通道时间序列输入
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    print(f"输入形状: {dummy_input.shape}")

    # 将数据送入模型
    output = model(dummy_input)
    print(f"最终输出形状: {output.shape}")

    # 验证输出形状是否符合预期
    expected_shape = torch.Size([BATCH_SIZE, OUTPUT_SEQ_LEN, OUT_CHANNELS])
    assert output.shape == expected_shape, f"形状不匹配! 期望得到 {expected_shape}, 但实际为 {output.shape}"

    print(f"\n成功! 模型输出了期望的形状: {expected_shape}")