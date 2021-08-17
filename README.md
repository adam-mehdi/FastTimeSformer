---

<div align="center">    
 
# FastTimeSformer

</div>
 
## Description   
FastTimeSformer adapts the divided space-time attention presented in TimeSformer to the linear-complexity attention approximation of FAVOR+. Code is drawn from repositories by @lucidrains.

## How to use   
```python
# intall project   
!pip install https://github.com/adam-mehdi/FastTimeSformer.git

from fast_timesformer.fast_timesformer import FastTimeSformer
```

You can use the model like any `nn.Module`:

```python
model = FastTimeSformer(...)
```

Or, you can use the fast space-time attention layer directly:

```python
attn = FastSpacetimeAttention(...)
```

### Citations
```
@misc{choromanski2021rethinking,
      title={Rethinking Attention with Performers}, 
      author={Krzysztof Choromanski and Valerii Likhosherstov and David Dohan and Xingyou Song and Andreea Gane and Tamas Sarlos and Peter Hawkins and Jared Davis and Afroz Mohiuddin and Lukasz Kaiser and David Belanger and Lucy Colwell and Adrian Weller},
      year={2021},
      eprint={2009.14794},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{bertasius2021spacetime,
      title={Is Space-Time Attention All You Need for Video Understanding?}, 
      author={Gedas Bertasius and Heng Wang and Lorenzo Torresani},
      year={2021},
      eprint={2102.05095},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

The code presented herein are adapted from two of @lucidrain's repositories:
1. [performer_pytorch](https://github.com/lucidrains/performer-pytorch)
2. [timesformer_pytorch](https://github.com/lucidrains/timesformer-pytorch)

```   
