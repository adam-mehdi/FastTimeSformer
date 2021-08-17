################################################# IMPORTS #####################################

import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from fast_timesformer.rotary import RotaryEmbedding, AxialRotaryEmbedding, apply_rot_emb
from fast_timesformer.helpers import linear_attention, gaussian_orthogonal_random_matrix, orthogonal_matrix_chunk, generalized_kernel, softmax_kernel, default, exists, PreNorm, FeedForward


################################################# ATTENTION ####################################################

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out
  
# slow attention for class token
def class_attention(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out
  
# Attention backbone
class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        """
        Given q, k, v of shape `(B, n_heads, n_tokens, head_dim)`, compute attention approximation a la Performers.

        Args:
            dim_head (int): size of each head in the multi-head attention
            nb_features (int): number of gausian orthogonal random features omega. 
                               Default: `int(dim_heads * math.log(dim_heads)`.
        """
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        "Shape: (B, heads - gh, N, dim_heads)"
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True) # b*n*h, f, head_dim
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out

# Fast Space-time attention layer
class FastSpacetimeAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.class_attn = class_attention
        self.fast_attn = FastAttention(dim_head)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, cls_mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale

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
      

################################################################### MODEL ############################

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
        rotary_emb = True
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

        NOTE: `forward` consumes videos of shape `(b, f, c, h, w)`.
        """
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

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
                PreNorm(dim, FastSpacetimeAttention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FastSpacetimeAttention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout))
            ]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video, mask = None):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        # calculate num patches in height and width dimension, and number of total patches (n)

        hp, wp = (h // p), (w // p)
        n = hp * wp

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
        
