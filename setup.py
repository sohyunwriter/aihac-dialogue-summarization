#nsml:hyeinhyun/first:sec

from setuptools import setup, find_packages
#from distutils.core import setup

install_requires = [
    "transformers",
    "pytorch-lightning",
    "torch",
    "deepspeed",
    "hydra-core",
    "scikit-learn",
    "tqdm",
]


setup(
    name="dialogue_summarization",
    install_requires=install_requires,
    packages=find_packages(),
    package_data={},
)
