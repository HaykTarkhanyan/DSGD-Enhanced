#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='cdsgd',
    version='0.1',
    author='Ricardo Valdivia',
    description='Tabular interpretable clustering based on Dempster-Shafer\
                 Theory and Gradient Descent',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ricardo-valdivia/CDSGD',
    packages=['cdsgd'],
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'dill',
        'dsgd @ git+https://github.com/Sergio-P/DSGD.git'
    ],
)
