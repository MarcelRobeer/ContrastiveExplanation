import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by ContrastiveExplanation (Foil Trees).')

setup(name='ContrastiveExplanation',
      version='0.1',
      python_requires='>3.6',
      description='Contrastive and counterfactual explanations for machine learning (ML) using Foil Trees',
      url='https://github.com/MarcelRobeer/ContrastiveExplanation',
      author='Marcel Robeer',
      author_email='m.j.robeer@uu.nl',
      license='BSD 3-Clause License',
      packages= find_packages(),
      install_requires=[
        'lime',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn>=0.18',
        'scikit-image>=0.12',
      ],
      classifiers = [
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
      ])