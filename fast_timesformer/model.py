################################################# IMPORTS #######################################################

import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from fast_timesformer.rotary import apply_rot_emb, AxialRotaryEmbedding, RotaryEmbedding, gaussian_orthogonal_random_matrix, softmax_kernel

################################################# HELPERS #######################################################

# normalizing
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
        
# feedforward
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# drop path
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


################################### REGULAR (QUADRATIC COMPLEXITY) ATTENTION #########################################

class RegularAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, qkv_bias=False, qk_scale = None, attn_drop = 0., proj_drop = 0., **kwargs):
        """
        Applies regular, quadratic complexity multi-head attention. 

        Args:
            dim (int): dimension of each token
            dim_head (int): size of each head for the multi-head attention
            qkv_bias (bool): whether to include bias in to_qkv projection
            qk_scale (float): scaling factor in softmax; defaults to 1/sqrt(head_dim)
            attn_drop (float): dropout rate applied to the attention distribution
            proj_drop (float): probability of zeroing each element at the end
        """
        super().__init__()
        assert dim % dim_head == 0, f'dim {dim} must be a multiple of dim_head {dim_head}'
        self.scale = qk_scale or dim_head**-.5
        self.dim_head = dim_head
        self.heads = dim // dim_head

        self.to_qkv = nn.Linear(dim, 3*dim, bias = qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)

        self.to_out = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        

    def forward(self, x, rot_emb = None):
        # create q, k, v with heads
        qo, ko, vo = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(partial(rearrange, pattern = 'b n (h d) -> (b h) n d', d = self.dim_head), (qo, ko, vo))

        # add rotary embeddings, if applicable
        if rot_emb is not None: q, k = apply_rot_emb(q, k, rot_emb)
        
        # compute attention distribution
        sim = torch.einsum('bid, bjd -> bij', q, k) * self.scale
        attn = self.attn_drop(sim.softmax(dim = -1))

        # calculate interaction between attn distribution and values, and project out
        res = rearrange(attn @ v, '(b h) n d -> b n (h d)', h = self.heads)
        return self.proj_drop(self.to_out(res))


######################################### PERFORMER (FAVOR) ATTENTION ####################################################
  
class PerformerAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, nb_features = None, ortho_scaling = 0, kernel_fn = nn.ReLU(), qkv_bias = False, **kwargs):
        """
        Given q, k, v of shape `(B, n_heads, n_tokens, head_dim)`, compute attention approximation a la Performers.

        Args:
            dim_head (int): size of each head in the multi-head attention
            nb_features (int): number of gausian orthogonal random features omega. 
                               Default: `int(dim_heads * math.log(dim_heads)`.
            ortho_scaling (int): scaling factor for the orthogonal random features
            kernel_fn (nn.Module): defines function to approximate with FAVOR+ framework
            qkv_bias (bool): whether to include bias in to_qkv projection
        """
        super().__init__()
        nb_features = nb_features if nb_features is not None else int(dim_head * math.log(dim_head))

        self.dim_head = dim_head
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_head, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.kernel_fn = kernel_fn

        self.to_qkv = nn.Linear(dim, 3*dim, bias = qkv_bias)
        self.to_out = nn.Linear(dim, dim)


    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, x, rot_emb = None):
        # create q, k, v with heads
        qo, ko, vo = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(partial(rearrange, pattern = 'b n (h d) -> (b h) n d', d = self.dim_head), (qo, ko, vo))

        # add rotary embeddings, if applicable
        if rot_emb is not None: q, k = apply_rot_emb(q, k, rot_emb)
        
        # create Q' and K' such that Q' @ K'.T approximates Softmax(Q @ K.T)
        q, k, v = map(partial(rearrange, pattern = '(b h) n d -> b h n d', b = x.shape[0]), (q, k, v))
        create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = q.device)
        q_prime = create_kernel(q, is_query = True) 
        k_prime = create_kernel(k, is_query = False)

        # multiply out D_inv @ (Q' @ (K' @ V)) where D_inv is the scaling factor
        k_cumsum = k.sum(dim = -2)
        D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)

        out = rearrange(out, 'b h n d -> b n (h d)', d=self.dim_head)
        return self.to_out(out)

################################################## FASTFORMER ATTENTION ############################

class FastformerAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, proj_drop = 0., qkv_bias = False, qk_scale = None, **kwargs):
        """
        Applies linear complexity fast attention from the fastformer paper.

        Args:
            dim (int): dimension of each token
            dim_head (int): size of each head for the multi-head attention
            drop (float): probability of zeroing each element at the end
            qkv_bias (bool): whether to include bias in to_qkv projection
            qk_scale (float): scaling factor in softmax; defaults to 1/sqrt(head_dim)
        """

        super().__init__()
        assert dim % dim_head == 0, f'dim {dim} must be a multiple of dim_head {dim_head}'
        self.scale = qk_scale or dim_head**-.5
        self.dim_head = dim_head
        self.heads = dim // dim_head

        self.to_qkv = nn.Linear(dim, 3*dim, bias = qkv_bias)
        self.Wq = nn.Linear(dim_head, 1, bias = False)
        self.Wk = nn.Linear(dim_head, 1, bias = False)

        self.to_out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p=proj_drop)
        

    def forward(self, x, rot_emb = None):
        # create q, k, v with heads
        qo, ko, vo = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(partial(rearrange, pattern = 'b n (h d) -> (b h) n d', d = self.dim_head), (qo, ko, vo))

        # add rotary embeddings, if applicable  
        if rot_emb is not None: q, k = apply_rot_emb(q, k, rot_emb)
                
        # compute query attention logits and global query vector
        alpha = F.softmax(self.Wq(q) * self.scale, dim = -2)
        q_global = reduce(alpha * q, 'b n d -> b 1 d', 'sum')

        # calculate key attention logits and global key vector
        p = q_global * k
        beta = F.softmax(self.Wk(p) * self.scale, dim = -2)
        k_global = reduce(beta * p, 'b n d -> b n 1', 'sum') 

        # compute key-value interaction and output transformation
        u = rearrange(k_global * v, '(b h) n d -> b n (h d)', h = self.heads)
        r = self.drop(self.to_out(u))
        return r + qo

################################################# TIMESFORMER BLOCK #########################################

class TimeSformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        frames,
        height_patches,
        width_patches,
        dim_head = 64,
        heads = 8,
        drop_path = 0.1,
        attn_drop = 0.,
        proj_drop = 0.,
        qkv_bias = False,
        mlp_ratio = 4,
        qk_scale = None,
        rotary_emb = True,
        attn_mechanism = 'fastformer',
        attn_type = 'divided',
    ):
        """
        Divided space-time attention with the capacity of linear complexity using
        fastformer or performer attention approximations.

        Args: 
            dim (int): size of each token in the input sequence `x`. Equal 
                            to `channels * patch_size**2`.
            frames (int): number of frames of the input video clip
            height_patches (int): number of patches in the height, height // patch_size
            width_patches (int): number of patches in the width, width // patch_size
            dim_head (int): size of each head for the multi-head self-attention applied
            heads (int): number of heads for the multi-head self-attention applied
            drop_path (float): rate of DropPath
            mlp_ratio (float): how much to scale `dim` for the hidden dim of MLP
            attn_drop (float): dropout probability in the attention layers
            proj_drop (float): dropout rate in feed forward layer and attention output fc
            qkv_bias (bool): whether the projection of `x` to qkv will include bias term
            qk_scale (float): scaling factor in softmax; defaults to 1/sqrt(head_dim)
            rotary_emb (bool): whether to use RoPE rotary embeddings in attention
            attn_mechanism (str): type of accelerated divided attention to use. 
                            Options are 'fastformer', 'performer', or 'regular'.
            attn_type (str): determines whether to attend over space and time jointly
                            or in a divided fashion. Options are 'joint' or 'divided'.
            

        Note: input `x` is of shape (b, seq_length + 1, patch_dim) where `+ 1`
                is due to the class token prepended to the sequence.

        """
        super().__init__()

        # assert arguments are valid
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        assert attn_mechanism in ['fastformer', 'performer', 'regular'], 'invalid attn mechanism option'
        assert attn_type in ['joint', 'divided'], 'invalid option for attn_type'

        self.frames, self.height_patches, self.width_patches, self.heads = frames, height_patches, width_patches, heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.attn_type = attn_type

        attn_types = {'performer': PerformerAttention, 'fastformer': FastformerAttention, 'regular': RegularAttention}
        attn_args = dict(dim=dim, dim_head=dim_head, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.attn_s = PreNorm(dim, attn_types[f'{attn_mechanism}'](**attn_args))

        self.mlp = PreNorm(dim, FeedForward(dim=dim, dropout=proj_drop, mult=mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # temporal attention parameters
        if attn_type == 'divided':
            self.attn_t = PreNorm(dim, attn_types[f'{attn_mechanism}'](**attn_args))
            self.fc_t = nn.Linear(dim, dim)


    def forward(self, x, frame_rot_emb = None, image_rot_emb = None, **kwargs):
        h, w, f = self.height_patches, self.width_patches, self.frames

        # attend across time and space jointly
        if self.attn_type == 'joint':
            x = self.drop_path(self.attn_s(x))
            x = self.drop_path(self.mlp(x))
            return x

        # divided spacetime attention
        else:
            ## Temporal 
            # remove classification token prepending each sequence in the batch
            init_cls_token, xt = x[:, :1, :], x[:, 1:, :]
            rep_cls_token = repeat(init_cls_token, 'b 1 m -> (b f) 1 m', f=f)

            # rearrange so that we attend across time
            res_t = rearrange(xt, 'b (h w f) m -> (b h w) f m', h=h, f=f)

            # multi-head attention across time
            res_t = self.drop_path(self.attn_t(res_t, rot_emb = frame_rot_emb))

            # rearrange, linear projection, and residual connection
            res_t = rearrange(res_t, '(b h w) t m -> b (h w t) m', h=h, w=w)
            out_t = self.fc_t(res_t) + xt

            ## Spatial
            # rearrange and concat the class token
            xs = rearrange(out_t, 'b (h w t) m -> (b t) (h w) m', h=h, w=w)
            xs = torch.cat((rep_cls_token, xs), 1)

            # attend over space
            res_s = self.drop_path(self.attn_s(xs, rot_emb = image_rot_emb))

            # average across frames for the cls token
            cls_token = reduce(res_s[:,0,:], '(b f) m -> b 1 m', 'mean', f=f)

            # rearrange output of spatial attention
            out_s = rearrange(res_s[:,1:,:], '(b f) (h w) m -> b (h w f) m', f=f, h=h)

            # residual connection
            x = torch.cat((init_cls_token, out_t), 1) + torch.cat((cls_token, out_s), 1)

            ## Feedforward
            x = x + self.drop_path(self.mlp(x))
            return x      

###################################################### FASTTIMESFORMER ######################################

class FastTimeSformer(nn.Module):
    def __init__(
        self,
        frames,
        channels,
        height, 
        width,
        num_classes,
        dim = 128,
        patch_size = 16,
        depth = 12,
        heads = 8,
        dim_head = 64,
        drop_path = .1,
        attn_dropout = 0.,
        proj_dropout = 0.,
        rotary_emb = True,
        attn_mechanism = 'fastformer',
        attn_type = 'divided',
        frames_first = True
    ):
        """
        Video transformer generalized to videos. Approximates divided space-time attention
        with the linearly-scalable FAVOR+ method.

        Args:
            frames (int): number of frames in each batch
            channels (int): number of channels in each frame's image
            height (int): height of each frame in the clip
            width (int): width of each frame in the clip
            num_classes (int): number of classes in the classification task. If binary
                            classification, `num_classes` should be 1.
            dim (int): size of each token when the clip is converted into a flat sequence
            patch_size (int): number of elements of each chunk into which the image is divided
                            in attention
            depth (int): amount of transformer blocks to use. A transformer block consists
                            of a time attention, spatial attention, and feed-forward layer
            heads (int): number of heads for the multi-head self-attention applied
            dim_head (int): size of each head for the multi-head self-attention applied
            drop_path (float): rate of DropPath
            attn_dropout (float): probability of zeroing out each element in the output 
                            of each attention layer.
            proj_dropout (float): probability of zeroing out each element in the output of 
                            each feed-forward layer and the output fc of attention.
            rotary_emb (bool): whether to use relative positional encodings using the RoPE
                            technique.
            attn_mechanism (str): type of accelerated divided attention to use. Options are 
                            'fastformer', 'performer', or 'regular'.
            attn_type (str): determines whether to attend over space and time jointly
                            or in a divided fashion. Options are 'joint' or 'divided'.
            frames_first (bool): whether `forward` consumes video clips of shape `(b, f, c, h, w)`.
                            if true or of shape `(b, c, f, h, w)` if false.

        NOTE: 
        """
        super().__init__()

        # init parameters
        f, c, h, w = frames, channels, height, width
        self.heads, self.patch_size = heads, patch_size
        assert h % patch_size == 0 and w % patch_size == 0, f'height {h} and width {w} must be divisible by patch size {patch_size}'
        
        self.hp, self.wp = (h // patch_size), (w // patch_size)
        seq_length = f * self.hp * self.wp
        patch_dim = c * (patch_size ** 2)

        # positional embeddings
        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(seq_length + 1, dim)
        
        # model parameters
        self.project_in = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))
        self.to_out = PreNorm(dim, nn.Linear(dim, num_classes))

        block_args = dict(
            dim=dim, frames=f, height_patches=self.hp, width_patches=self.wp, dim_head=dim_head, drop_path=drop_path, 
            attn_drop=attn_dropout, proj_drop=proj_dropout, attn_mechanism=attn_mechanism, rotary_emb=rotary_emb, attn_type=attn_type
            )
        self.blocks = nn.ModuleList([TimeSformerBlock(**block_args) for _ in range(depth)])
        self.frames_first = frames_first

    def forward(self, video, mask = None):
        b, f, c, h, w, device, p = *video.shape, video.device, self.patch_size
        
        if not self.frames_first: 
            f, c = c, f
            video = rearrange(video, 'b c f h w -> b f c h w')
            
        # video to patch embeddings
        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = p, p2 = p)
        tokens = self.project_in(video)

        # add cls token
        cls_token = repeat(self.cls_token, 'n d -> b n d', b = b)
        x =  torch.cat((cls_token, tokens), dim = 1)

        # positional embedding
        frame_pos_emb, image_pos_emb = None, None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device = device)
            image_pos_emb = self.image_rot_emb(self.hp, self.wp, device = device)

        # blocks constitute time attn, space attn, then feedforward layer
        for block in self.blocks: 
            x = block(x, frame_rot_emb = frame_pos_emb, image_rot_emb = image_pos_emb)

        # extract class token and project to output
        cls_token = x[:, 0]
        return self.to_out(cls_token)
        
        
