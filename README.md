---

<div align="center">    
 
# FastTimeSformer

</div>
 
The TimeSformer is an attention-based video classifier. FastTimeSformer adds to it accelerated attention approximations. It provides three options: `fastformer`, `performer` and `regular`, all of which support enhanced relative position embeddings.

## Options

- `fastformer` implements the linear complexity additive attention mechanism presented in the recent paper, "FastFormer: Additive Attention Can Be All You Need" (Aug 2021)
<additive attn mechanism>

 - `performer` provides the linear complexity attention approximation achieved via FAVOR+, a method of sampling orthogonal random features, presented in "Rethinking Attention with Performers" (Sept 2020)

 - `regular` is the standard quadratic complexity attention used in the TimeSformer Paper, "Is Space-Time Attention All You Need for Video Understanding?" (Feb 2021)

## How to use   
```python
# intall project   
!pip install git+https://github.com/adam-mehdi/FastTimeSformer.git

import torch
from fast_timesformer.model import FastTimeSformer

b, f, c, h, w = 16, 5, 3, 128, 128
x = torch.randn(b, f, c, h, w)
model = FastTimeSformer(...)

model(x)
```

Or you can use the fast attention layer directly:

```python
from fast_timesformer.model import DividedAttention


x = torch.randn(b, n*f + 1, dim) 

attention = DividedAttention(dim, dim_head = dim_head, heads = heads, dropout = dropout)

time_attended = attention(x, 'b (f n) d', '(b n) f d', n = n)        # attention across frames
space_attended = attention(x, 'b (f n) d', '(b f) n d', f = f)       # attention across patches
```

## Citations
```
@misc{choromanski2021rethinking,
      title={Rethinking Attention with Performers}, 
      author={Krzysztof Choromanski and Valerii Likhosherstov and David Dohan and Xingyou Song 
              and Andreea Gane and Tamas Sarlos and Peter Hawkins and Jared Davis and Afroz Mohiuddin 
              and Lukasz Kaiser and David Belanger and Lucy Colwell and Adrian Weller},
      year={2021},
      eprint={2009.14794},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{wu2021fastformer,
      title={Fastformer: Additive Attention Can Be All You Need}, 
      author={Chuhan Wu and Fangzhao Wu and Tao Qi and Yongfeng Huang and Xing Xie},
      year={2021},
      eprint={2108.09084},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{bertasius2021spacetime,
      title={Is Space-Time Attention All You Need for Video Understanding?}, 
      author={Gedas Bertasius and Heng Wang and Lorenzo Torresani},
      year={2021},
      eprint={2102.05095},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
 
@misc{su2021roformer,
      title={RoFormer: Enhanced Transformer with Rotary Position Embedding}, 
      author={Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
      year={2021},
      eprint={2104.09864},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

The code here was built on top of a couple repositories by lucidrains:
1. [performer_pytorch](https://github.com/lucidrains/performer-pytorch)
2. [timesformer_pytorch](https://github.com/lucidrains/timesformer-pytorch)

```   
