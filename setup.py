#!/usr/bin/env python

from setuptools import setup

meta = {
    'name': 'scikit-gof',
    'version': '0.1.3',
    'packages': ('skgof',),
    'install_requires': (
        'numpy>=1.10',
        'scipy>=0.16'
    ),
    'tests_require': (
        'flake8-print',
        'flake8-todo',
        'pep8-naming',
        'pytest',
        'pytest-benchmark',
        'pytest-flake8',
        'pytest-isort',
        'pytest-readme'
    ),
    'description': "Variations on goodness of fit tests for SciPy.",
    'long_description': open('README.rst').read(),
    'author': "Wojciech Ruszczewski",
    'author_email': "scipy@wr.waw.pl",
    'url': "https://github.com/wrwrwr/scikit-gof",
    'license': "MIT",
    'classifiers': (
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5"
    )
}

if __name__ == '__main__':
    setup(**meta)
