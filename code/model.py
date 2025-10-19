from functools import partial
import tqdm
import torch
import torch.nn as nn
import os


# ---------------------- 1. 带空洞卷积的图像分块嵌入模块 ----------------------
class patched(nn.Module):
    def __init__(self, imgsize=224, patch_size=16, in_channels=3, embed_dim=512,
                 norm_layer=None):  # 移除dilation参数（深度可分离卷积无需空洞率）
        super().__init__()
        img_size = (imgsize, imgsize)
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.img_size = img_size

        # 深度可分离卷积无需空洞率，直接计算普通卷积的填充（确保输出尺寸与原分块逻辑一致）
        # 目标：输出特征图尺寸 = (img_size - patch_size) / stride + 1 = (224-16)/16 +1 =14x14
        padding_h = (patch_size[0] - 1) // 2  # 普通卷积填充公式（保持边缘信息）
        padding_w = (patch_size[1] - 1) // 2
        self.padding = (padding_h, padding_w)

        # 计算分块网格大小（必须为14x14，与原逻辑一致）
        self.grid_size = (
            (img_size[0] + 2 * padding_h - patch_size[0]) // patch_size[0] + 1,
            (img_size[1] + 2 * padding_w - patch_size[1]) // patch_size[1] + 1
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 14*14=196，确保与原模型兼容

        # ---------------------- 核心修改：深度可分离卷积替换原卷积 ----------------------
        # 1. 深度卷积（Depth-wise Convolution）：逐通道独立卷积，保持通道数不变
        #    groups=in_channels 表示每个输入通道用独立卷积核处理
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # 输出通道数=输入通道数（仅逐通道卷积）
            kernel_size=patch_size,
            stride=patch_size,  # 步长=patch_size，确保分块数量不变
            padding=self.padding,
            groups=in_channels  # 关键：深度卷积的分组数=输入通道数
        )

        # 2. 逐点卷积（Point-wise Convolution）：1x1卷积融合通道，映射到目标嵌入维度
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,  # 输入=深度卷积的输出通道数（in_channels）
            out_channels=embed_dim,  # 输出=目标嵌入维度（512）
            kernel_size=1,  # 1x1卷积，仅融合通道
            stride=1,
            padding=0
        )

        # 组合深度卷积和逐点卷积为proj（替代原单一卷积）
        self.proj = nn.Sequential(
            self.depth_conv,
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            self.point_conv
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f'输入图像大小{H}*{W}与模型期望{self.img_size[0]}*{self.img_size[1]}不匹配'

        # 深度可分离卷积分块+展平：输出形状仍为 (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)  # 与原逻辑完全兼容
        x = self.norm(x)
        return x

# ---------------------- 2. 注意力模块（保持不变） ----------------------
class attention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super(attention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------- 3. MoE-MLP模块（修复专家选择逻辑） ----------------------
class MoE_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_fn=nn.GELU, dropout=0.1, num_experts=8, top_k=2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4  # 专家隐藏层维度

        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(in_features, num_experts)  # 门控网络：预测专家权重

        # 专家列表：多个独立MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                act_fn(),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, out_features),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, N, C = x.shape  # (B, 197, 512)
        x_flat = x.reshape(B * N, C)  # 展平为 (B*N, C)，便于门控计算

        # 门控选择top-k专家
        gate_weights = self.gate(x_flat)  # (B*N, num_experts)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # (B*N, top_k)
        top_k_weights = top_k_weights.softmax(dim=-1)  # 权重归一化

        # 核心修复：按选中的专家ID动态选择专家（而非固定取前k个）
        final_output = torch.zeros_like(x_flat)  # 存储最终输出
        for i in range(self.top_k):
            expert_ids = top_k_indices[:, i]  # 第i个选中的专家ID（B*N个）
            weights = top_k_weights[:, i].unsqueeze(1)  # 对应的权重（B*N, 1）

            # 遍历所有专家，处理被选中的样本
            for expert_idx in range(self.num_experts):
                # 找到选择当前专家的样本掩码
                mask = (expert_ids == expert_idx)
                if mask.any():
                    # 对选中的样本应用当前专家
                    expert_output = self.experts[expert_idx](x_flat[mask])
                    final_output[mask] += expert_output * weights[mask]

        # 恢复原形状
        return final_output.reshape(B, N, C)


# ---------------------- 4. Transformer块（集成注意力和MoE-MLP） ----------------------
class block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., act_fn=nn.GELU, norm_layer=nn.RMSNorm,
                 num_experts=8, top_k=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention(
            embed_dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MoE_MLP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_fn=act_fn, dropout=drop, num_experts=num_experts, top_k=top_k
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # 注意力残差
        x = x + self.mlp(self.norm2(x))  # MoE-MLP残差
        return x


# ---------------------- 5. 完整Vision Transformer（带空洞卷积和MoE） ----------------------
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3, num_classes=100, embed_dim=256,
                 depth=6, num_heads=4, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, norm_layer=nn.RMSNorm, attn_drop=0, drop_ratio=0,
                 embed_layer=patched, act_layer=None, num_experts=8, top_k=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_tokens = 1  # 仅用cls_token
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # 图像分块嵌入（使用带空洞卷积的embed_layer）
        self.embed_layer = embed_layer(
            imgsize=img_size, patch_size=patch_size, in_channels=in_channel,
            embed_dim=embed_dim, norm_layer=norm_layer, # 传递空洞率
        )
        num_patches = self.embed_layer.num_patches  # 应保持196

        # 类别token和位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_ratio)

        # Transformer编码器（含MoE）
        self.blocks = nn.Sequential(*[
            block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_ratio, act_fn=act_layer, norm_layer=norm_layer,
                attn_drop=attn_drop, num_experts=num_experts, top_k=top_k
            ) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # 分类头
        self.pre_logits = nn.Identity()  # 无额外预分类层
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def forward_features(self, x):
        x = self.embed_layer(x)  # 分块嵌入：(B, 196, 512)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, 512)
        x = torch.cat((cls_token, x), dim=1)  # (B, 197, 512)
        x = self.pos_drop(x + self.pos_embed)  # 加位置嵌入
        x = self.blocks(x)  # Transformer编码
        x = self.norm(x)
        return self.pre_logits(x[:, 0])  # 取cls_token

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.RMSNorm):
            nn.init.ones_(m.weight)
        elif isinstance(m, MoE_MLP):
            # MoE门控层单独初始化
            nn.init.trunc_normal_(m.gate.weight, std=0.01)
            if m.gate.bias is not None:
                nn.init.zeros_(m.gate.bias)


# ---------------------- 6. 模型构建函数（指定空洞率和MoE参数） ----------------------
def vit_base_patch16_224_moe(num_experts,top_k,classes=100,img_size=224, patch_size=16, in_channel=3,
        embed_dim=512, depth=4, num_heads=8, mlp_ratio=2.0):
    return VisionTransformer(
        img_size=224, patch_size=16, in_channel=3, num_classes=classes,
        embed_dim=512, depth=4, num_heads=8, mlp_ratio=2.0,
        num_experts=num_experts, top_k=top_k,
    )





