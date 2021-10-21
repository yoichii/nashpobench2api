from setuptools import find_packages, setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nashpobench2api',
    packages=find_packages(),
    version='1.0.0',
    description='API for NAS-HPO-Bench-II.',
    author='Yoichi Hirose',
    license='MIT',
    install_requires=[],
    url='https://github.com/yoichii/nashpobench2api',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
