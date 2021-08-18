#!/usr/bin/env python

from setuptools import setup, find_packages


  
from setuptools import setup, find_packages

setup(
  name = 'fast_timesformer',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'TimeSformer accelerated with linearly-scaling attention',
  author = 'Adam Mehdi',
  author_email = 'adam.mehdi23@gmail.com',
  url = 'https://github.com/adam-mehdi/FastTimeSformer.git',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'video classification',
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
)
