from setuptools import setup, find_packages

# get directory of setup.py
# import os
# setup_dir = os.path.dirname(os.path.abspath(__file__))

# get requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nba_pistons',
    version='0.2.0',
    author='Alexander Hernandez',
    author_email='ahernandezjr0@gmail.com',
    description='A package for collecting and analyzing NBA player data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ahernandezjr/nba_pistons',
    packages=find_packages(
        where='src',
        include=['*'],
    ),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)