from setuptools import setup, find_packages

setup(
    name='torchid',
    version='0.1',
    url='https://github.com/forgi86/pytorch-ident.git',
    author='Marco Forgione',
    author_email='marco.forgione1986@gmail.com',
    description='System identification with pytorch package',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'torch'],  # to be checked
    extras_require={
        'continuous-time integration': ["nodepy"],
        'download datasets': ["requests"],
        'open datasets': ["pandas"],
        'generate documentation': ["sphinx"]
    }
)
