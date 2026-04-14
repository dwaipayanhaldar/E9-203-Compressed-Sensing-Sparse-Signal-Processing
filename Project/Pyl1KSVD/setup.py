from setuptools import setup, find_packages

setup(
    name='pyl1ksvd',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'matplotlib',
        'Pillow',
    ],
    author='Dwaipayan Haldar',
    description=(
        'A Python implementation of the l1-K-SVD robust dictionary learning '
        'algorithm (Mukherjee, Basu, Seelamantula, Signal Processing 2016).'
    ),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
