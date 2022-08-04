from setuptools import setup


setup(
    name='sil-v0',
    version='0.0.1',
    packages=[
        'ENVS',
        'ENVS.envs',
        'ENVS.envs.configs',
        'ENVS.envs.policy',
        'ENVS.envs.utils',
],
    install_requires=[
        'gitpython',
        'gym',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
