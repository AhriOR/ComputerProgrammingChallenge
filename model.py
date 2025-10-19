from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

class patched(nn.Module):
    def __init__(self,imgsize=224,patch_size=16,in_channels=3,embed_dim=768,norm_layer=None):
        super(self).__init__()
        img_size=(imgsize,imgsize)
        patch_size=(patch_size,patch_size)
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size=(img_size[0]//patch_size[0],img_size[1]//patch_size[1])
        self.num_patches=self.grid_size[0]*self.grid_size[1]

        self.proj = nn.Conv2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm=norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):
        B,C,H,W=x.shape
        assert H==self.img_size[0] and W==self.img_size[1],\
        f'输入图像大小{H}*{W}与模型期望不匹配'
        #(B,3,224,224)->B,768,14,14->B,196,768
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class attention(nn.Module):
    def __init__(self,embed_dim=768,num_heads=8,qkv_bias=False,qk_scale=None, # 缩放因子 1/sqrt(embed_dim_pre_head)
                 attn_drop=0., #注意力分数dropout比例
                 proj_drop=0.):#最终投影层dropout比例
        super(attention, self).__init__()

        self.num_heads=num_heads
        head_dim=self.embed_dim//num_heads #每个注意力头的维度
        self.scale = qk_scale or head_dim**-0.5
        self.qkv=nn.Linear(embed_dim,embed_dim*3,bias=qkv_bias)# 通过全连接层生成qkv，为了并行计算，提高计算效率
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj_drop=nn.Dropout(proj_drop)
        # 将每个head得到的输出进行concat拼接，然后通过线性变换映射回原本的嵌入dim
        self.proj = nn.Linear(embed_dim,embed_dim,bias=qkv_bias)

    def forward(self,x):
        B,N,C=x.shape #batch,num_patches+1(class_token),embed_dim
        #B N 3*C -> B N 3 num_heads,C//num_heads
        #B N 3 num_heads,C//num_heads-> 3,B,num_heads,N,C//num_heads
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2, 0, 3, 1, 4)
        #B num_heads N C//num_heads
        q,k,v=qkv[0],qkv[1],qkv[2]
        #得到注意力分数 B,num_heads N N
        attn=(q @ k.transpose(-2,-1)) * self.scale
        attn=attn.softmax(dim=-1) #对每行进行处理
        # 注意力权重对v进行加权求和
        # B,num_heads,N,N(v的维度) @ B,num_heads N C//num_heads ->B num_heads N C//num_heads
        #->B N num_heads C//num_heads ->B,N,C
        x=(attn @ v).transpose(1,2).reshape(B,N,C)
        x=self.proj(x)
        x=self.attn_drop(x)

        return x

class mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_fn=nn.GELU,dropout=0.1):
        #in_feature 输入维度hidden_features 隐藏层维度，通常为in——features四倍
        super().__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.fc1=nn.Linear(in_features,hidden_features)
        self.ac1=act_fn()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        x=self.fc1(x)
        x=self.ac1(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.dropout(x)
        return x

class block(nn.Module):
    def __init__(self,dim,num_heads=8,mlp_ratio=4.0,qkv_bias=False, qk_scale=None,drop=0.,attn_drop=0.,act_fn=nn.GELU,norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1=norm_layer(dim)
        self.attn=attention(embed_dim=dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
        self.norm2=norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp=mlp(in_features=dim,hidden_features=mlp_hidden_dim,out_features=mlp_hidden_dim,act_fn=act_fn,dropout=drop)

    def forward(self,x):
        x=x+self.attn(self.norm1(x))
        x=x+self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_channel=3,num_classes=1000,embed_dim=768,depth=12,num_heads=8,mlp_ratio=4.0,qkv_bias=True,qk_scale=None,representation_size=None,distil=False,norm_layer=nn.LayerNorm,
                 attn_drop=0,drop_ratio=0,embed_layer=patched,act_layer=None,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.num_heads=num_heads
        self.num_features=self.embed_dim=embed_dim
        self.num_classes=num_classes
        self.num_tokens= 2 if distil else 1
        norm_layer=norm_layer or partial(nn.LayerNorm,eps=1e-6)
        act_layer=act_layer or nn.GELU
        self.embed_layer=embed_layer(imgsize=img_size,patch_size=patch_size,in_channels=in_channel,embed_dim=embed_dim,norm_layer=norm_layer)
        num_patches = self.embed_layer.num_patches
        self.cls_token=nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token=nn.Parameter(torch.zeros(1, 1, embed_dim))
        # B 197 768
        self.pos_embed=nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop=nn.Dropout(drop_ratio)
        #使用nn.sequential将列表中所有模块打包为一个整体
        self.block=nn.Sequential(*[
            block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop_ratio,act_fn=act_layer,norm_layer=norm_layer,attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm=norm_layer(embed_dim)

        if representation_size and not distil:
            self.has_logits=True
            self.num_features=representation_size
            # 对cls_token做映射
            self.pre_logits=nn.Sequential(OrderedDict([
                ('fc',nn.Linear(embed_dim,representation_size)),
                ('act',nn.Tanh())
            ]))
        else:
            self.has_logits=False
            self.pre_logits=nn.Identity() #pre_logits不做处理

        self.head = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist =None
        if distil:
            self.head_dist = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()

        #权重初始化
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token,std=0.02)

        nn.init.trunc_normal_(self.cls_token,std=0.02)
        self.apply(self._init_weights)

    def forward_features(self,x):
        # B C H W->B num_patches embed_dim
        x = self.embed_layer(x)
        # 1 1 768 ->B 1 768
        cls_token=self.cls_token.expand(x.shape[0],1,-1)
        #disk_token 蒸馏标记
        if self.dist_token is None:
            # B 197 768
            x=torch.cat((cls_token,x),dim=1)
        else:
            # B 198 768
            x=torch.cat((cls_token,self.dist_token.expand(x.shape[0],1,-1),x),dim=1)
        x=self.pos_drop(x+self.pos_embed)
        x=self.block(x)
        x=self.norm(x)
        if self.dist_token is None:
            #cls token
            return self.pre_logits(x[:,0])
        else:
            return x[:,0],x[:,1]
    def forward(self,x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x,x_dist=self.head(x[0]),self.head_dist(x[1])
            # 如果是训练模式且不是脚本模式
            if self.training and torch.jit.is_scripting():
                return x,x_dist

        else:
            x=self.head(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight,std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def vit_base_patch16_224(classes=1000,pretrained=False,**kwargs):
    model = VisionTransformer(img_size=224,patch_size=16,in_channel=3,num_classes=classes,embed_dim=768,depth=12,num_heads=8,representation_size=None,distil=False,norm_layer=nn.LayerNorm)
    return model
