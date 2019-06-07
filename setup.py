"""Setup module."""
from pip._internal import main as pip_main  # noqa
from setuptools import find_packages, setup

setup(
    name='moa-python',
    version='0.0.1',
    url='https://github.com/abdelrahman-t/moa-python',
    author='abdelrahman-t',
    author_email='abdurrahman.talaat@gmail.com',
    description=('Pythonic wrapper around Massive Online Analysis MOA'),
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6.0',
    install_requires=['py4j']
)
