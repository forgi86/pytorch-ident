# System identification tools in PyTorch
A collection of system identification tools implemented in PyTorch.

* State-space identification methods (see [1], [2], [3], [6])
* Differentiable transfer functions (see [4], [5])

## Examples and Documentation

* Examples are provided in the [**examples**](examples) folder of this repo.
* The API documentation is available at https://pytorch-ident.readthedocs.io/en/latest.


## Installation:

### Requirements:
A Python 3.9 conda environment with

 * numpy
 * scipy
 * matplotlib
 * pandas
 * pytorch
 
### Stable version from PyPI

Run the command 

```
pip install pytorch-ident
```
This will install the current [stable version](https://pypi.org/project/pytorch-ident/) from the PyPI package repository.

### Latest version from GitHub
1. Get a local copy the project. For instance, run 
```
git clone https://github.com/forgi86/pytorch-ident.git
```
in a terminal to clone the project using git. Alternatively, download the zipped project from [this link](https://github.com/forgi86/pytorch-ident/zipball/master) and extract it in a local folder

2. Install pytorch-ident by running
```
pip install .
```
in the project root folder (where the file setup.py is located). 

# Bibliography
[1] M. Forgione and D. Piga. Model structures and fitting criteria for system identification with neural networks. In Proceedings of the 14th IEEE International Conference Application of Information and Communication Technologies, 2020. <br/><br/>
[2] B. Mavkov, M. Forgione, D. Piga. Integrated Neural Networks for Nonlinear Continuous-Time System Identification. IEEE Control Systems Letters, 4(4), pp 851-856, 2020. <br/><br/>
[3] M. Forgione and D. Piga. Continuous-time system identification with neural networks: model structures and fitting criteria. European Journal of Control, 59:68-81, 2021. <br/><br/>
[4] M. Forgione and D. Piga. dynoNet: a neural network architecture for learning dynamical systems. International Journal of Adaptive Control and Signal Processing, 2021. <br/><br/>
[5] D. Piga, M.Forgione and M. Mejari. Deep learning with transfer functions: new applications in system identification. In Proceedings of the the 2021 SysId Conference, 2021. <br/><br/>
[6] G. Beintema, R. Toth and M. Schoukens. Nonlinear state-space identification using deep encoder networks. Learning for Dynamics and Control. PMLR, 2021. <br/><br/>
