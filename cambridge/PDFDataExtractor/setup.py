#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()

setup(
    name='pdfdataextractor',
    version='1.0',
    author='Miao Zhu',
    author_email='zhumiao@hotmail.co.uk',
    license='MIT',
    url='https://github.com/cat-lemonade/PDFDataExtractor',
    packages=find_packages(),
    description='A toolkit for automatically extracting semantic information from PDF files of scientific articles.',
    long_description='A toolkit for automatically extracting semantic information from PDF files of scientific articles.',
    keywords='text-mining pdf science scientific',
    zip_safe=False,
    entry_points={},
    tests_require=['pytest'],
    install_requires=['pdfminer.six==20200121'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
    ],
)
