from setuptools import setup, find_packages

setup(
    name='pytorch-ident',
    version='0.2.1',
    url='https://github.com/forgi86/pytorch-ident.git',
    author='Marco Forgione',
    author_email='marco.forgione1986@gmail.com',
    description='System identification with pytorch package',
    packages=["torchid"],
    install_requires=['numpy', 'scipy', 'matplotlib', 'torch'],  # to be checked
    extras_require={
        'download datasets': ["requests", "googledrivedownloader"],
        'open datasets': ["pandas"],
        'generate plots': ['matplotlib'],
        'generate documentation': ["sphinx"]
    }
)
