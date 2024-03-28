from setuptools import setup, find_packages

setup(
    name='torch_ac_simple',
    version='0.1.34',
    author='Stevo Huncho',
    author_email='stevo@stevohuncho.com',
    description='Simple implementation of A2C and PPO using PyTorch',
    keywords="reinforcement learning, actor-critic, a2c, ppo, multi-processes, gpu",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch_ac',
        'tensorboardX',
    ],
)