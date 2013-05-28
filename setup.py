# setup.py
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Create the setup dict
setup_args = {
    'name' : 'metric-learn',
    'version' : '0.0.0',
    'author' : 'John Collins',
    'author_email' : 'johnssocks@gmail.com',
    'packages' : ['metric_learn', 'metric_learn.test'],
    'scripts' :[],
    'url' : 'github.com/johncollins/metric-learn',
    'license' :'LICENSE.txt',
    'description' : 'Learn mahalanobis style metrics parameterized by some learned matrix A',
    'long_description' : open('README.md', 'r').read(),
    'requires' : ['numpy']
}

setup(**setup_args)
