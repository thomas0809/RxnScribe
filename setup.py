from distutils.core import setup
from pathlib import Path


def get_install_requires():
    """Returns requirements.txt parsed to a list"""
    fname = Path(__file__).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets


requirements = get_install_requires()

setup(name='RxnScribe',
      version='1.0',
      description='RxnScribe',
      author='Yujie Qian',
      author_email='yujieq@csail.mit.edu',
      url='https://github.com/Ozymandias314/MolDetect',
      packages=['rxnscribe', 'rxnscribe.inference', 'rxnscribe.pix2seq', 'rxnscribe.transformer'],
      package_dir={'rxnscribe': 'rxnscribe'},
      package_data={},
      setup_requires=['numpy'],
      install_requires=requirements,
      dependency_links=['git+https://github.com/thomas0809/MolScribe.git@main#egg=MolScribe'],
      )
