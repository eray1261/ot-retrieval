from setuptools import setup, find_packages

# get the requirements from the requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()
setup(
    name='STRAP',
    version='1.0',
    packages=find_packages(),
    install_requires=required,
)
