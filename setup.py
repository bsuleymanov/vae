import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['vae']
from version import __version__

setup(
  name = 'vae',
  packages = find_packages(),
  entry_points={
    'console_scripts': [
      'vae = vae.cli:main',
    ],
  },
  version = __version__,
  license='MIT',
  description = 'VAE',
  author = 'Bulat Suleymanov',
  author_email = 'motjumi@gmail.com',
  url = 'https://github.com/bsuleymanov/vae',
  keywords = [
    'variational autoencoder'
  ],
  install_requires=[
    'fire',
    'torch>=1.6',
    'torchvision',
  ],
)