---

<div align="center">    
 
# FastTimeSformer

</div>
 
The TimeSformer is an attention-based video classifier. FastTimeSformer adds to it accelerated attention approximations. It provides three options: `fastformer`, `performer` and `regular`, all of which support enhanced relative position embeddings.

I provide extensive documentation in the docstrings... Enjoy!

## Options

- `fastformer` implements the linear complexity additive attention mechanism presented in the recent paper, "FastFormer: Additive Attention Can Be All You Need" (Aug 2021)
<additive attn mechanism>
![](https://dingyue.ws.126.net/2021/0827/887f9828p00qyhdsf0018d200u000gvg00zk00jz.png)
 
 - `performer` provides the linear complexity attention approximation achieved via FAVOR+, a method of sampling orthogonal random features, presented in "Rethinking Attention with Performers" (Sept 2020)
 
![](https://1.bp.blogspot.com/-pQ8s4X2qXjI/X5Ib6nLtxWI/AAAAAAAAGtI/C7dmMqV3Gu0NGYtmi5Gqjkr_Pqun5T2MwCLcBGAsYHQ/s1428/image10.jpg) 

 - `regular` is the standard quadratic complexity attention used in the TimeSformer Paper, "Is Space-Time Attention All You Need for Video Understanding?" (Feb 2021)
 
## Relative performance
 
The following graph measures the runtime of a forward pass through TimeSformer using 

- Fastformer outperforms regular attention at mid-to-large image resolutions. Note too that fastformer 
- If you are working with small images or with few frames, using regular attention is optimal.
- Performer is not performing. It seems creating the softmax kernel is taking longer than computing the actual quadratic attention map, even for these large inputs. This part of the code is from lucidrain's performer-pytorch repository

 
## How to use   
```python
# intall project   
!pip install git+https://github.com/adam-mehdi/FastTimeSformer.git

import torch
from fast_timesformer.model import FastTimeSformer

b, f, c, h, w = 16, 5, 3, 512, 512
x = torch.randn(b, f, c, h, w)
model = FastTimeSformer(f, c, h, w, num_classes = 2, attn_mechanism = 'fastformer')

model(x)
```

You can also use the fast attention layers directly for a different model:

```python
from fast_timesformer.model import FastAttention, RegularAttention

b, n, d = 16, 32, 64
x = torch.randn(b, n, d) 
fast_attn = FastAttention(dim = d)
reg_attn = RegularAttention(dim = d)
 
attended_fastformer = fast_attn(x)
attended_regular = reg_attn(x)
```

## Citations

Papers referenced:
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
 
```
 
Portions of the code was built on top of a couple repositories by lucidrains:
1. [performer-pytorch](https://github.com/lucidrains/performer-pytorch)
2. [timesformer-pytorch](https://github.com/lucidrains/timesformer-pytorch)

