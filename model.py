from collections import OrderedDict
from functools import partial
import tqdm
import torch
import torch.nn as nn
from torch import optim

from dataloader import train_dataloader


class patched(nn.Module):
    def __init__(self, imgsize=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (imgsize, imgsize)
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f'输入图像大小{H}*{W}与模型期望不匹配'
        # (B,3,224,224)->B,768,14,14->B,196,768
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class attention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super(attention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads  # 每个注意力头的维度
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=qkv_bias)  # 并行生成qkv
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)  # 注意力输出投影

    def forward(self, x):
        B, N, C = x.shape  # batch, num_patches+1(cls_token), embed_dim
        # B N 3*C -> B N 3 num_heads C//num_heads -> 3 B num_heads N C//num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离q、k、v

        # 计算注意力分数并加权
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力输出拼接与投影
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_fn=nn.GELU, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4  # 修正：MLP隐藏层默认设为输入4倍（符合ViT标准）
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ac1 = act_fn()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., act_fn=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention(embed_dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_fn=act_fn, dropout=drop)

    def forward(self, x):
        # 残差连接：注意力层
        x = x + self.attn(self.norm1(x))
        # 残差连接：MLP层
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    # 关键修改1：移除distil参数（不再需要蒸馏标记）
    def __init__(self, img_size=224, patch_size=16, in_channel=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 representation_size=None, norm_layer=nn.LayerNorm, attn_drop=0, drop_ratio=0,
                 embed_layer=patched, act_layer=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.num_heads = num_heads
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        # 关键修改2：num_tokens固定为1（仅保留cls_token）
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # 图像分块与嵌入
        self.embed_layer = embed_layer(imgsize=img_size, patch_size=patch_size,
                                       in_channels=in_channel, embed_dim=embed_dim, norm_layer=norm_layer)
        num_patches = self.embed_layer.num_patches

        # 关键修改3：仅定义cls_token，删除dist_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置嵌入：num_patches + 1（仅cls_token）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_ratio)

        # Transformer编码器块
        self.block = nn.Sequential(*[
            block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_ratio, act_fn=act_layer, norm_layer=norm_layer, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # 预分类层（可选）
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # 关键修改4：删除head_dist（仅保留主分类头）
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def forward_features(self, x):
        # 1. 图像分块嵌入：B C H W -> B num_patches embed_dim
        x = self.embed_layer(x)
        # 2. 拼接cls_token：B num_patches embed_dim -> B (num_patches+1) embed_dim
        # 关键修改5：仅拼接cls_token，删除dist_token拼接逻辑
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 扩展到batch维度
        x = torch.cat((cls_token, x), dim=1)
        # 3. 位置嵌入与dropout
        x = self.pos_drop(x + self.pos_embed)
        # 4. Transformer编码器前向传播
        x = self.block(x)
        x = self.norm(x)
        # 5. 提取cls_token输出（第0个token），并通过预分类层
        # 关键修改6：仅返回cls_token的处理结果，无其他token
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        # 关键修改7：删除head_dist分支，仅处理cls_token的分类输出
        x = self.forward_features(x)
        x = self.head(x)
        return x

    # 权重初始化方法（不变）
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


# 模型构建函数
def vit_base_patch16_224(classes=100, pretrained=False, **kwargs):
    # 关键修改8：创建模型时移除distil=False参数（模型__init__已无该参数）
    model = VisionTransformer(img_size=224, patch_size=16, in_channel=3, num_classes=classes,
                             embed_dim=768, depth=12, num_heads=8, representation_size=None,
                             norm_layer=nn.LayerNorm, **kwargs)
    return model


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 初始化模型
model = vit_base_patch16_224(pretrained=False).to(device)
# 数据加载器（来自dataloader.py）
training_loader = train_dataloader
# 优化器
optimizer = optim.AdamW(model.parameters(), weight_decay=0.0001, lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
# 训练参数
num_epochs = 50
criterion = nn.CrossEntropyLoss().to(device)  # 分类任务损失


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()  # 模型设为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        # tqdm进度条
        loop = tqdm.tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True
        )
        for batch_idx, (images, labels) in loop:
            # 数据移至设备
            images = images.to(device)
            labels = labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播：仅输出cls_token的分类结果（无dist_token）
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播与参数更新
            loss.backward()
            optimizer.step()
            # 损失统计与进度条更新
            running_loss += loss.item()
            batch_avg_loss = running_loss / (batch_idx + 1)
            loop.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{batch_avg_loss:.4f}"
            })
        # 打印当前epoch平均损失
        epoch_avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} 完成，平均损失: {epoch_avg_loss:.4f}\n")


# 启动训练
if __name__ == "__main__":
    train_model(
        model=model,
        train_loader=training_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )