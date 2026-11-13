import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# 假设您的所有模块都在这些文件中
# 请确保 AFDEE.py, ST_IWAE.py, TimesNet.py, 和 ns_Transformer.py 在同一目录下
from networks.MyNetWork.AFDEE import UnifiedTSLANet
from networks.MyNetWork.ST_IWAE import ST_IWAE
from networks.MyNetWork.TimesNet import CNN_BiLSTM_Classifier
from networks.MyNetWork.ns_Transformer import NSTransformerEncoder  # 导入新的模块四
from networks.MyNetWork.ResNet import ResNet1DEncoder


# =================================================================================
# 1. HSIC 损失函数 (新增)
# 这是实现两个分支解耦的关键
# =================================================================================

def rbf_kernel(x, y, sigma=1.0):
    """计算RBF核矩阵"""
    # x, y: [batch_size, features]
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def hsic_loss(X, Y, sigma=1.0):
    """
    计算HSIC损失来度量X和Y的独立性。
    损失值越小，表示X和Y越独立。
    """
    m = X.shape[0]
    if m < 2:
        return torch.tensor(0.0).to(X.device)

    # 创建中心化矩阵 H
    H = torch.eye(m, device=X.device) - 1.0 / m

    # 计算核矩阵
    Kx = rbf_kernel(X, X, sigma)
    Ky = rbf_kernel(Y, Y, sigma)

    # 计算HSIC
    hsic = torch.trace(Kx @ H @ Ky @ H) / ((m - 1) ** 2)
    return hsic


# =================================================================================
# 2. 重新设计的双分支模型
# =================================================================================

class ActionSubjectDisentanglementModel(nn.Module):
    """
    一个双分支解耦模型，用于同时进行动作识别和个体识别。

    - 分支1 (动作识别): AFDEE -> ST_IWAE -> Classifier
    - 分支2 (个体识别): ResNet1DEncoder -> Classifier  <--- 修改
    - 解耦机制: 通过HSIC损失函数最小化两个分支核心表征的依赖性。
    """

    def __init__(self,
                 # 动作分支 - AFDEE 参数
                 afdee_in_seq_len, afdee_out_seq_len, afdee_in_chans, afdee_out_chans,
                 afdee_patch_size, afdee_embed_dim, afdee_num_blocks, afdee_drop_rate,
                 # 动作分支 - ST_IWAE 参数
                 st_iwae_seq_len, st_iwae_in_channels, st_iwae_latent_dim_t, st_iwae_latent_dim_s,
                 # 动作分支 - 分类器参数
                 action_classifier_in_channels, action_num_classes, action_cnn_out,
                 action_lstm_hidden, action_lstm_layers,
                 # 个体分支 - NSTransformer 参数
                 subject_in_channels, subject_seq_len, subject_encoder_features,
                 # 个体分支 - 分类器参数
                 subject_num_classes,
                 # ===== AFDEE 消融相关开关 =====
                 use_afdee: bool = True,
                 afdee_use_spectral: bool = True,
                 afdee_use_icb: bool = True,
                 afdee_use_pos_embed: bool = True,
                 afdee_use_backbone: bool = True,
                 afdee_spectral_mode: str = "afdee"):
        super().__init__()

        # --- 分支1: 动作识别 ---
        self.use_afdee = use_afdee
        self.afdee_out_seq_len = afdee_out_seq_len  # 用于插值到 256
        self.afdee_spectral_mode = afdee_spectral_mode

        if self.use_afdee:
            # AFDEE 频域增强模块（可通过 afdee_spectral_mode 控制三种策略）
            self.afdee = UnifiedTSLANet(
                in_seq_len=afdee_in_seq_len,
                out_seq_len=afdee_out_seq_len,
                in_chans=afdee_in_chans,
                out_chans=afdee_out_chans,
                patch_size=afdee_patch_size,
                embed_dim=afdee_embed_dim,
                num_blocks=afdee_num_blocks,
                drop_rate=afdee_drop_rate,
                # 内部消融/策略
                use_spectral=afdee_use_spectral,
                use_icb=afdee_use_icb,
                use_pos_embed=afdee_use_pos_embed,
                use_backbone=afdee_use_backbone,
                spectral_mode=afdee_spectral_mode,
            )
            self.afdee_adapter = None
        else:
            #  w/o AFDEE：不用 AFDEE，只用一个非常简单的 1x1 Conv + 线性插值
            self.afdee = None
            self.afdee_adapter = nn.Conv1d(
                in_channels=afdee_in_chans,  # 8
                out_channels=afdee_out_chans,  # 8
                kernel_size=1
            )

        self.st_iwae = ST_IWAE(
            seq_len=st_iwae_seq_len, in_channels=st_iwae_in_channels,
            latent_dim_t=st_iwae_latent_dim_t, latent_dim_s=st_iwae_latent_dim_s
        )
        self.action_classifier = CNN_BiLSTM_Classifier(
            in_channels=action_classifier_in_channels, num_classes=action_num_classes,
            cnn_out_channels=action_cnn_out, lstm_hidden_size=action_lstm_hidden,
            lstm_num_layers=action_lstm_layers
        )

        # --- 分支2: 个体识别 (已修改为ResNet) ---
        # 修正了原先的重复定义问题，并替换为ResNet1DEncoder
        self.subject_encoder = NSTransformerEncoder(
            in_channels=subject_in_channels,
            seq_len=subject_seq_len,
            output_features=subject_encoder_features
        )
        # self.subject_encoder = ResNet1DEncoder(
        #     in_channels=subject_in_channels,
        #     output_features=subject_encoder_features
        #     # 这里可以使用ResNet1DEncoder的默认配置，也可以传入更多参数进行定制
        # )
        self.subject_classifier = nn.Linear(subject_encoder_features, subject_num_classes)

    def forward(self, x):
        """
        定义完整的前向传播路径。
        Args:
            x (torch.Tensor): 原始输入时序数据, shape [B, L, C]。
        """
        # --- 分支1: 动作识别传播 (保持不变) ---
        # --- 分支1: 动作识别传播 ---
        if self.use_afdee:
            # 正常：AFDEE 频域增强 + TSLANet
            embedded_x = self.afdee(x)  # [B, 256, 8]
        else:
            # 🔴 w/o AFDEE：只做简单的通道线性变换 + 时间插值
            # x: [B, 400, 8] -> [B, 8, 400]
            x_t = x.transpose(1, 2)
            # 1x1 Conv：只在通道上做线性变换，不动时间维度
            x_t = self.afdee_adapter(x_t)  # [B, 8, 400]
            # 线性插值到 256 个时间点
            x_t = F.interpolate(x_t, size=self.afdee_out_seq_len,
                                mode='linear', align_corners=False)  # [B, 8, 256]
            embedded_x = x_t.transpose(1, 2)  # [B, 256, 8]

        # 后面 ST_IWAE 和分类器保持不变

        x_reconstructed, mu_t, logvar_t, mu_s, logvar_s, z_t, z_s = self.st_iwae(embedded_x)
        action_logits = self.action_classifier(x_reconstructed)

        # 动作分支的核心表征 (用于HSIC解耦)
        action_features = torch.cat([z_t, z_s], dim=1)

        # --- 分支2: 个体识别传播 (现在使用ResNet) ---
        subject_features = self.subject_encoder(x)
        subject_logits = self.subject_classifier(subject_features)

        return (action_logits, subject_logits,
                x_reconstructed, mu_t, logvar_t, mu_s, logvar_s,
                embedded_x, action_features, subject_features)


# =================================================================================
# 3. 示例用法与测试代码 (已更新)
# =================================================================================
if __name__ == "__main__":
    # --- 1. 定义超参数 ---
    BATCH_SIZE = 16
    RAW_SEQ_LEN = 400
    RAW_CHANNELS = 8
    ACTION_CLASSES = 37
    NUM_SUBJECTS = 10
    AFDEE_OUT_SEQ_LEN = 256
    AFDEE_OUT_CHANS = 8
    LAMBDA_HSIC = 0.5

    # --- 2. 实例化双分支模型 ---
    model = ActionSubjectDisentanglementModel(
        # 动作分支 - AFDEE
        afdee_in_seq_len=RAW_SEQ_LEN, afdee_out_seq_len=AFDEE_OUT_SEQ_LEN,
        afdee_in_chans=RAW_CHANNELS, afdee_out_chans=AFDEE_OUT_CHANS,
        afdee_patch_size=16, afdee_embed_dim=128, afdee_num_blocks=5, afdee_drop_rate=0.2,
        # 动作分支 - ST_IWAE
        st_iwae_seq_len=AFDEE_OUT_SEQ_LEN, st_iwae_in_channels=AFDEE_OUT_CHANS,
        st_iwae_latent_dim_t=64, st_iwae_latent_dim_s=64,
        # 动作分支 - 分类器
        action_classifier_in_channels=AFDEE_OUT_CHANS, action_num_classes=ACTION_CLASSES,
        action_cnn_out=128, action_lstm_hidden=128, action_lstm_layers=2,
        # 个体分支 - NSTransformer
        subject_in_channels=RAW_CHANNELS, subject_seq_len=RAW_SEQ_LEN, subject_encoder_features=128,
        # 个体分支 - ResNet1D <--- 修改
        # subject_in_channels=RAW_CHANNELS, subject_encoder_features=128,
        # 个体分支 - 分类器
        subject_num_classes=NUM_SUBJECTS,
        use_afdee=True
    )
    model.train()
    print("--- 双分支解耦模型实例化成功 ---")
    print(f"总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M\n")

    # --- 3. 设置优化器和所有损失函数 (保持不变) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_action_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()


    def loss_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    criterion_subject_cls = nn.CrossEntropyLoss()
    criterion_hsic = hsic_loss

    # --- 4. 创建模拟数据和标签 (保持不变) ---
    dummy_input = torch.randn(BATCH_SIZE, RAW_SEQ_LEN, RAW_CHANNELS)
    dummy_action_labels = torch.randint(0, ACTION_CLASSES, (BATCH_SIZE,))
    dummy_subject_labels = torch.randint(0, NUM_SUBJECTS, (BATCH_SIZE,))
    print(f"--- 输入和标签的形状 ---")
    print(f"{'输入形状:':<25} {dummy_input.shape}")
    print(f"{'动作标签形状:':<25} {dummy_action_labels.shape}")
    print(f"{'个体标签形状:':<25} {dummy_subject_labels.shape}\n")

    # --- 5. 执行一个完整的训练步骤 (保持不变) ---
    try:
        print("--- 执行前向和后向传播 ---")
        optimizer.zero_grad()
        (action_logits, subject_logits,
         x_rec, mu_t, logvar_t, mu_s, logvar_s,
         embedded_x, action_features, subject_features) = model(dummy_input)

        # 验证重构损失的维度是否匹配
        if x_rec.shape != embedded_x.shape:
            print(
                f"Warning: Shape mismatch for reconstruction loss. x_rec: {x_rec.shape}, embedded_x: {embedded_x.shape}")

        loss_act_cls = criterion_action_cls(action_logits, dummy_action_labels)
        loss_recon = criterion_recon(x_rec, embedded_x)
        loss_kld = loss_kl(mu_t, logvar_t) + loss_kl(mu_s, logvar_s)
        loss_subj_cls = criterion_subject_cls(subject_logits, dummy_subject_labels)
        loss_hsic_val = criterion_hsic(action_features, subject_features)
        total_loss = (loss_act_cls + loss_recon + 0.1 * loss_kld +
                      loss_subj_cls + LAMBDA_HSIC * loss_hsic_val)
        total_loss.backward()
        optimizer.step()
        print("前向和后向传播成功完成。\n")

        # --- 6. 验证所有输出的形状和损失值 (保持不变) ---
        print("--- 验证输出形状和损失值 ---")
        print(f"{'动作 Logits 形状:':<25} {action_logits.shape}")
        print(f"{'个体 Logits 形状:':<25} {subject_logits.shape}")
        print(f"{'动作表征形状 (HSIC):':<25} {action_features.shape}")
        print(f"{'个体表征形状 (HSIC):':<25} {subject_features.shape}")
        print("-" * 50)
        print(f"{'动作分类损失:':<25} {loss_act_cls.item():.4f}")
        print(f"{'个体分类损失:':<25} {loss_subj_cls.item():.4f}")
        print(f"{'重构损失:':<25} {loss_recon.item():.4f}")
        print(f"{'KL 散度损失:':<25} {loss_kld.item():.4f}")
        print(f"{'HSIC 解耦损失:':<25} {loss_hsic_val.item():.4f}")
        print(f"{'总复合损失:':<25} {total_loss.item():.4f}")

    except Exception as e:
        print(f"\n--- 训练步骤中发生错误 ---")
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()