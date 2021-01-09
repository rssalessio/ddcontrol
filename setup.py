from setuptools import setup
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name = 'ddcontrol',
    version = '0.0.1',
    description = 'Data-Driven Control Python Library',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    keywords = ['Data Driven Control', 'VRFT', 'Virtual Reference Feedback Tuning',
                'Adaptive Control'],
    url = 'https://github.com/rssalessio/ddcontrol/',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    license = 'MIT',
    packages = ['ddcontrol', 'test', 'examples'],
    zip_safe = False,
    install_requires = [
        'scipy',
        'numpy',
    ],
    test_suite = 'nose.collector',
    test_requires = ['nose'],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering"
    ],
    python_requires = '>=3.5',
)
