################################################# IMPORTS #######################################################

import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from fast_timesformer.rotary import RotaryEmbedding, AxialRotaryEmbedding, apply_rot_emb


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



######################################### PERFORMER (FAVOR) ATTENTION #############################################
  
# regular attention for class token and if specified
class RegularAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, q, k, v):
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        scale = q.shape[-1]**-.5
        attn = F.Softmax(sim * scale, dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return out
  
class PerformerAttention(nn.Module):
    def __init__(self, dim_heads = 64, nb_features = None, ortho_scaling = 0, kernel_fn = nn.ReLU(), **kwargs):
        """
        Given q, k, v of shape `(B, n_heads, n_tokens, head_dim)`, compute attention approximation a la Performers.

        Args:
            dim_head (int): size of each head in the multi-head attention
            nb_features (int): number of gausian orthogonal random features omega. 
                               Default: `int(dim_heads * math.log(dim_heads)`.
        """
        super().__init__()
        nb_features = nb_features if nb_features is not None else int(dim_heads * math.log(dim_heads))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.kernel_fn = kernel_fn


    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        "Shape: (B, heads - gh, N, dim_heads)"
        device = q.device
        
        create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
        q = create_kernel(q, is_query = True) 
        k = create_kernel(k, is_query = False)

        out = linear_attention(q, k, v)
        return out

################################################## FASTFORMER ATTENTION ######################################

class FastformerAttention3d(nn.Module):
    def __init__(self, dim, n, dim_head = 64, heads = 8, dropout = 0.):
        """
        Applies linear complexity fast attention from the fastformer paper.

        Args:
            dim (int): dimension of each token
            n (int): size of each input sequence
            dim_head (int): size of each head for the multi-head attention
            heads (int): number of heads to for the multi-head attention

        NOTE: Input `x` is of shape (bs, n, dim)
        """
        super().__init__()

        self.scale = dim_head**-.5
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, *3)
        self.Wq = nn.Linear(dim_head, n)
        self.Wk = nn.Linear(dim_head, n)
        self.fc_out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        

    def forward(self, x):
        # create queries, keys, values
        qo, ko, vo = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(partial(rearrange, pattern = 'b n (h d) -> b h n d', d = self.dim_head), (qo, ko, vo))

        # create global query vector
        A = F.softmax(self.Wq(q) * self.scale, dim = -1)
        q_global = reduce(A @ q, 'b h n d -> b h n 1', 'sum')

        # use global query vector to create global key vector
        p = q_global*k
        B = F.softmax(self.Wk(p) * self.scale, dim = -1)
        k_global = reduce(B @ p, 'b h n d -> b h n 1', 'sum')

        # element-wise product global query & vector and query residual-like addition
        u = rearrange(k_global*v, 'b h n d -> b n (h d)')
        out = self.to_out(u) + qo
        return out

################################################# DIVIDED ATTENTION #########################################

# 3d attention backbone
class DividedAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        attn_type = 'fastformer'
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.class_attn = RegularAttention()
        attn_types = {'performer': PerformerAttention, 'fastformer': FastformerAttention, 'regular': RegularAttention}
        self.fast_attn = attn_types['attn_type'](dim_head)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, cls_mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out = self.class_attn(cls_q, k, v, mask = cls_mask)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        q_, k_, v_ = map(lambda t: rearrange(t, '(b h) f d -> b h f d', h = h), (q_, k_, v_))
        out = self.fast_attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, 'b h f d -> (b h) f d', h=h)
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim = 1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)
      

######################################################## TIMESFORMER ######################################

class FastTimeSformer(nn.Module):
    def __init__(
        self,
        *,
        num_frames,
        num_classes,
        dim = 128,
        image_size = 224,
        patch_size = 16,
        channels = 3,
        depth = 12,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        rotary_emb = True,
        
    ):
        """
        Video transformer generalized to videos. Approximates divided space-time attention
        with the linearly-scalable FAVOR+ method.

        Args:
            num_frames (int): number of frames in each batch
            num_classes (int): number of classes in the classification task
            dim (int): size of each token
            image_size (int): image height or width. Only square images are supported.
            patch_size (int): height or width of each chunk into which the image is divided
            channels (int): number of channels in each frame's image
            depth (int): amount of transformer blocks to use. A transformer block consists
                            of time attention, spatial attention, and feed-forward layers
            heads (int): number of heads for the multi-head self-attention applied
            dim_head (int): size of each head 
            attn_dropout (float): probability of zeroing out each element in the output 
                            of each attention layer.
            ff_dropout (float): probability of zeroing out each element in the output of 
                            each feed-forward layer.
            rotary_emb (bool): whether to use relative positional encodings using the RoPE
                            technique.
            attention_type (str): type of accelerated divided attention to use. Options are
                            'fastformer', 'performer', or 'regular'.

        NOTE: `forward` consumes videos of shape `(b, f, c, h, w)`.
        """
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        
        (image_size // patch_size)**2

        self.heads = heads
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + 1, dim)


        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FastSpacetimeAttention(dim, dim_head = dim_head, num_frames = num_frames, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FastSpacetimeAttention(dim, dim_head = dim_head, num_frames = num_frames, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout))
            ]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video, mask = None):
        b, f, _, h, w, *_, device, p, n = *video.shape, video.device, self.patch_size, self.n_patches

        # video to patch embeddings
        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = p, p2 = p)
        tokens = self.to_patch_embedding(video)

        # add cls token
        cls_token = repeat(self.cls_token, 'n d -> b n d', b = b)
        x =  torch.cat((cls_token, tokens), dim = 1)

        # positional embedding
        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device = device)
            image_pos_emb = self.image_rot_emb(hp, wp, device = device)

        # calculate masking for uneven number of frames
        frame_mask = None
        cls_attn_mask = None
        if exists(mask):

            cls_attn_mask = repeat(mask, 'b f -> (b h) () (f n)', n = n, h = self.heads)
            cls_attn_mask = F.pad(cls_attn_mask, (1, 0), value = True)

        # time and space attention
        for (time_attn, spatial_attn, ff) in self.layers:
            # Note: Frame mask for `time_attn` removed so that we can apply fast attention.
            x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
            x = ff(x) + x

        cls_token = x[:, 0]
        return self.to_out(cls_token)
        
