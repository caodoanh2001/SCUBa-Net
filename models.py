import torch.nn.functional as F
import torch.utils.data as data
import torch
import timm
from gcn import GCNBlock
import torch.nn as nn
import numpy as np
from skimage.util.shape import view_as_windows
from att_utils import MultiheadAttention

class CrossAttention(nn.Module):
    """
    Cross-Attention between two branches. Originaly introduced in https://github.com/IBM/CrossViT
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MHSA(nn.Module):
    """
    Define a MHSA class for stacking in nn.ModuleList()
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mhsa = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)

    def forward(self, x, att_mask=None):
        return self.mhsa(x, x, x, attn_mask=att_mask)[0]

class SCUBaNet(nn.Module):
    def __init__(self, num_nodes=None, node_dim=None, embed_dim=None, num_gcns=12, bn=1, add_self=1, normalize_embedding=1, num_classes=4):
        super().__init__()
        """
        SCUBa-Net

        :param num_gcns: int: The number of GCN layers in F^C
        :param num_nodes: int: The number of nodes in spatially-constrained graph G^C
        :param node_dim: int, optional: The number of nodes in spatially-constrained graph G^C
        :param embed_dim: int, optional: Embedding size of nodes in both G_c G^C and G^U
        """
        self.num_gcns = num_gcns
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.bn = bn
        self.add_self = add_self
        self.normalize_embedding = normalize_embedding
        self.num_classes = num_classes

        # 1. DESIGN OF F^C BLOCK
        ## Virtual node of spatially-constrained graph
        self.vrt_node_c = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) 
        nn.init.normal_(self.vrt_node_c, std=1e-6)
        
        ## First N_L GCN layers to process spatially-constrained graph
        self.F_c_gcn = nn.ModuleList([
            GCNBlock(self.node_dim, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 'relu')
            if (i == 0) else \
            GCNBlock(self.embed_dim, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 'relu')
            for i in range(self.num_gcns)
        ])

        ## GCN layers to update virtual node of spatially-constrained graph
        self.F_c_update_vrt_node = nn.ModuleList([
            GCNBlock(self.embed_dim, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 'relu')
            for _ in range(self.num_gcns)
        ])

        ## Masked multi-head self-attention to capture long-range dependencies
        self.F_c_mhsa = nn.ModuleList(
            [
                MHSA(embed_dim=self.embed_dim, num_heads=8)
                for _ in range(self.num_gcns)
            ]
        )

        # 2. DESIGN OF F^U BLOCK
        ## F^{U} block is built upon Vision Transformer
        vit = timm.create_model('vit_base_patch32_384', num_classes=0, pretrained=True).cuda()
        self.vit_pre = vit.patch_embed
        self._pos_embed = vit._pos_embed
        self.vit_blocks = nn.ModuleList([*vit.blocks]).cuda()
        
        # 3. DESIGN OF F^{X}_{N->N} block
        vit_regions = view_as_windows(np.arange(0, 144).reshape(12, 12), (3, 3), step=3).reshape(-1, 9) # To view seq of ViT patches as a tensor 784 x 12 x 12
        self.u2c_relationships = torch.from_numpy(vit_regions).cuda() # Relationship indices
        
        ## Multi-head Cross-attention to transfer from nodes of spatially-unconstrained graph to spatially-constrained graph
        self.mhca_from_gu_to_gc = nn.ModuleList([
            CrossAttention(dim=self.embed_dim, num_heads=8)
            for _ in range(self.num_gcns)
        ])

        # 4. DESIGN OF F^{X}_{C->U}
        self.mhca_c_to_u = CrossAttention(dim=self.embed_dim, num_heads=8)

        # 5. DESIGN OF F^{X}_{U->C}
        self.mhca_u_to_c = CrossAttention(dim=self.embed_dim, num_heads=8)

        # Classifier
        self.classifier_ = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim * 1, self.num_classes),
        )

    def get_G_u_by_indices(self, tokens, indices):
        """
        To get indices of ViT tokens, i.e., nodes in spatially-unconstrained graph to interact with node in spatially-constrained graph

        :param tokens: torch.Tensor: Seq of ViT tokens
        :param indices: torch.Tensor: Indices as a tensor
        """
        patch_tokens = []
        for ind in indices:
            patch_tokens.append(tokens[:, ind, :].unsqueeze(1))
        return patch_tokens

    def forward(self, imgs, G_c, adjs):
        """

        :param G_c: torch.Tensor: Spatially-constrained graph
        :param G_u: torch.Tensor: Spatially-unconstrained graph
        """
        imgs = imgs.permute(0, 3, 1, 2)
        bs, _, _ = G_c.shape

        # 1. SPATIALLY-CONSTRAINED GRAPH'S BRANCH
        ## Adjs for spatially-constrained graph's nodes
        nodes_adjs = adjs[:, 1:, 1:]
        masks = torch.ones(G_c.shape[:2]).to(adjs)

        ## Update of spatially-constrained graph's nodes
        for node_layer in self.F_c_gcn:
            G_c = node_layer(G_c, nodes_adjs, masks)

        ## Concatenate virtual node node for spatially-constrained graph
        vrt_node_c = self.vrt_node_c.repeat(bs, 1, 1)
        G_c = torch.cat([vrt_node_c, G_c], dim=1)
        masks = torch.ones(G_c.shape[:2]).to(adjs)

        # 2. SPATIALLY-UNCONSTRAINED GRAPH
        G_u = self.vit_pre(imgs)
        G_u = self._pos_embed(G_u)

        ## 3. BI-GRAPH INTERACTION IMPLEMENTATION
        # [cls] node update and ViT-GCN interactions
        for _, (cls_layer, vit_block, cls_mhsa_layer, cross_att) in enumerate(zip(self.F_c_update_vrt_node, self.vit_blocks, self.F_c_mhsa, self.mhca_from_gu_to_gc)):
            # Vision transformer block
            G_u = vit_block(G_u)

            # Update virtual node along with other nodes in spatially-constrained graph
            G_c = cls_layer(G_c, adjs, masks)
            G_c += cls_mhsa_layer(G_c, att_mask=(adjs == 0).repeat(8, 1, 1))

            # Node-to-node interaction
            patch_G_u = self.get_G_u_by_indices(G_u[:, 1:, :], self.u2c_relationships)
            patch_G_u = torch.cat(patch_G_u, dim=1) # bs, 16, 9, 768
            patch_G_u = patch_G_u.view(-1, self.u2c_relationships.shape[-1], self.embed_dim) # bs * 16, 9, 768

            reshaped_G_c = G_c[:, 1:].reshape(-1, self.embed_dim).unsqueeze(1) # bs * 16, 1, 768
            relationship = reshaped_G_c + cross_att(torch.cat([reshaped_G_c, patch_G_u], dim=1))
            G_c[:, 1:] = relationship.reshape(bs, self.u2c_relationships.shape[0], self.embed_dim) # bs, 1, 768

        ## 4. U->C interaction
        G_c[:, 0:1] += self.mhca_c_to_u(torch.cat([G_c[:, 0:1, :], G_u[:, 1:]], dim=1))
        
        ## 5. C->U interaction
        G_u[:, 0:1] += self.mhca_u_to_c(torch.cat([G_u[:, 0:1, :], G_c[:, 1:]], dim=1))

        # 5. Classification using only virtual nodes
        ce_logits = [G_u[:, 0], G_c[:, 0]]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        probs = self.classifier_(ce_logits)
        
        # 6. We done!
        return probs
    
select_model = SCUBaNet