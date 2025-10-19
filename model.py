from collections import OrderedDict
from functools import partial
import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import os

# 导入数据加载器（假设包含训练集和测试集）
from dataloader import train_dataloader


# ---------------------- 1. 带空洞卷积的图像分块嵌入模块 ----------------------
class patched(nn.Module):
    def __init__(self, imgsize=224, patch_size=16, in_channels=3, embed_dim=512,
                 norm_layer=None, dilation=2):  # 新增dilation参数（空洞率）
        super().__init__()
        img_size = (imgsize, imgsize)
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.img_size = img_size
        self.dilation = dilation  # 空洞率（1=普通卷积，2=空洞卷积）

        # 关键计算：空洞卷积的有效核大小和填充（确保输出尺寸与普通卷积一致）
        # 有效核大小 = 物理核大小 + (物理核大小-1)*(dilation-1)
        effective_kernel_h = patch_size[0] + (patch_size[0] - 1) * (dilation - 1)
        effective_kernel_w = patch_size[1] + (patch_size[1] - 1) * (dilation - 1)

        # 计算填充：补偿空洞导致的尺寸缩减，确保输出网格大小仍为14x14
        padding_h = (effective_kernel_h - patch_size[0]) // 2
        padding_w = (effective_kernel_w - patch_size[1]) // 2
        self.padding = (padding_h, padding_w)

        # 计算分块网格大小（必须与原普通卷积一致，否则位置嵌入维度错误）
        self.grid_size = (
            (img_size[0] + 2 * padding_h - effective_kernel_h) // patch_size[0] + 1,
            (img_size[1] + 2 * padding_w - effective_kernel_w) // patch_size[1] + 1
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 应保持14*14=196

        # 空洞卷积替换普通卷积（核心修改）
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,  # 步长仍为patch_size，确保分块数量不变
            padding=self.padding,
            dilation=dilation  # 空洞率
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f'输入图像大小{H}*{W}与模型期望{self.img_size[0]}*{self.img_size[1]}不匹配'
        # 空洞卷积分块+展平：输出形状仍为 (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
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
    def __init__(self, img_size=224, patch_size=16, in_channel=3, num_classes=100, embed_dim=512,
                 depth=8, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 representation_size=None, norm_layer=nn.RMSNorm, attn_drop=0, drop_ratio=0,
                 embed_layer=patched, act_layer=None, num_experts=8, top_k=2, dilation=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_tokens = 1  # 仅用cls_token
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # 图像分块嵌入（使用带空洞卷积的embed_layer）
        self.embed_layer = embed_layer(
            imgsize=img_size, patch_size=patch_size, in_channels=in_channel,
            embed_dim=embed_dim, norm_layer=norm_layer, dilation=dilation  # 传递空洞率
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
def vit_base_patch16_224_moe(classes=100, num_experts=8, top_k=2, dilation=2, **kwargs):
    return VisionTransformer(
        img_size=224, patch_size=16, in_channel=3, num_classes=classes,
        embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0,
        num_experts=num_experts, top_k=top_k, dilation=dilation,  # 传递空洞率和MoE参数
        **kwargs
    )


# ---------------------- 7. 训练函数（含最优模型保存） ----------------------
def train_model(model, train_loader, criterion, optimizer, lr_scheduler, num_epochs, device,
                save_path='./models/best_model.pth'):
    model.train()
    best_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        current_lr = optimizer.param_groups[0]['lr']

        loop = tqdm.tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs} | LR: {current_lr:.6f}",
            leave=True
        )

        for batch_idx, (images, labels) in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 反向传播
            loss.backward()
            optimizer.step()

            # 更新进度条
            running_loss += loss.item()
            batch_avg_loss = running_loss / (batch_idx + 1)
            avg_acc = 100 * correct / total
            batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            loop.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{batch_avg_loss:.4f}",
                "batch_acc": f"{batch_acc:.2f}%",
                "avg_acc": f"{avg_acc:.2f}%"
            })

        # 调整学习率
        lr_scheduler.step()

        # 计算epoch指标
        epoch_avg_loss = running_loss / len(train_loader)
        epoch_avg_acc = 100 * correct / total

        # 保存最优模型
        if epoch_avg_acc > best_acc:
            best_acc = epoch_avg_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'loss': epoch_avg_loss
            }, save_path)
            print(f"✅ 最优模型已保存 | 最高准确率: {best_acc:.2f}% | 路径: {save_path}")

        print(
            f"\nEpoch {epoch + 1} 完成 | 平均损失: {epoch_avg_loss:.4f} | 平均准确率: {epoch_avg_acc:.2f}% | 历史最高: {best_acc:.2f}%\n")


# ---------------------- 8. 预测函数（测试集评估） ----------------------
def predict(model, test_loader, device, model_path=None):
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📥 加载模型 | 保存时准确率: {checkpoint['best_acc']:.2f}% | Epoch: {checkpoint['epoch']}")

    model.eval()
    all_preds = []
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
        for batch_idx, (images, labels) in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_acc = 100 * correct / total
            all_preds.extend(predicted.cpu().numpy().tolist())

            loop.set_postfix({"test_acc": f"{test_acc:.2f}%"})

    print(f"\n📊 测试完成 | 总样本: {total} | 准确率: {test_acc:.2f}%")
    return all_preds, test_acc


# ---------------------- 9. 主函数（启动训练和测试） ----------------------
if __name__ == "__main__":
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型（含空洞卷积和MoE）
    model = vit_base_patch16_224_moe(
        classes=100,
        num_experts=8,  # MoE专家数量
        top_k=2,  # 每个样本选择的专家数
        dilation=2  # 空洞率（1=普通卷积，2=空洞卷积）
    ).to(device)

    # 数据加载器
    train_loader = train_dataloader


    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.0001,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    # 学习率策略：5个epoch warmup + 45个epoch余弦退火
    warmup_epochs = 5
    main_epochs = 50 - warmup_epochs
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-6)
    lr_scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [warmup_epochs])

    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 启动训练
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=50,
        device=device,
        save_path='./models/best_moe_vit_dilated.pth'
    )

