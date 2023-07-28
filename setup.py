from distutils.core import setup

setup(
    name='RxnScribe',
    version='1.0',
    description='RxnScribe',
    author='Yujie Qian',
    author_email='yujieq@csail.mit.edu',
    url='https://github.com/thomas0809/RxnScribe',
    packages=['rxnscribe', 'rxnscribe.inference', 'rxnscribe.pix2seq', 'rxnscribe.transformer'],
    package_dir={'rxnscribe': 'rxnscribe'},
    package_data={},
    setup_requires=['numpy'],
    install_requires=[
        'torch',
        'numpy>=1.19.5',
        'pandas>=1.2.4',
        'Pillow==9.5.0',
        'matplotlib>=3.5.3',
        'opencv-python>=4.5.5.64',
        'pycocotools>=2.0.4',
        'pytorch-lightning>=1.8.6',
        'transformers>=4.5.1',
        'huggingface-hub>=0.11.0',
        'MolScribe @ git+https://github.com/thomas0809/MolScribe.git@main#egg=MolScribe',
        'easyocr>=1.6.2',
    ],
    dependency_links=['git+https://github.com/thomas0809/MolScribe.git@main#egg=MolScribe'],
)
