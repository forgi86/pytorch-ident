from setuptools import setup

setup(
    name='pytorch-ident',
    version='0.2.2',
    url='https://github.com/forgi86/pytorch-ident.git',
    author='Marco Forgione',
    author_email='marco.forgione1986@gmail.com',
    description='A library for system identification with PyTorch',
    packages=["torchid"],
    install_requires=['numpy', 'scipy', 'matplotlib', 'torch'],  # to be checked
    extras_require={
        'download datasets': ["requests", "googledrivedownloader"],
        'open datasets': ["pandas"],
        'generate plots': ['matplotlib'],
        'generate documentation': ["sphinx"]
    }
)
