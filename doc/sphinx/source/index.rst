.. pytorch-ident documentation master file, created by
   sphinx-quickstart on Fri Apr 10 01:50:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pytorch-ident
===========================
------------------------------------------------
System Identification with  PyTorch.
------------------------------------------------

pytorch-ident is an open-source python library for system identification
with deep learning tools based on the PyTorch framework.
The project is hosted on this `GitHub repository <https://github.com/forgi86/pytorch-ident>`_.

Requirements
------------

In order to run pytorch-ident, you need a python 3.x environment and the following packages:

* `PyTorch <https://pytorch.org/>`_
* `numpy <https://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_
* `matplotlib <https://matplotlib.org/>`_

Installation
------------
1. Copy or clone the pytorch-ident project into a local folder. For instance, run

.. code-block:: bash

   git clone https://github.com/forgi86/pytorch-ident.git

from the command line

2. Navigate to your local pyMPC folder

.. code-block:: bash

   cd <LOCAL_FOLDER>

where <LOCAL_FOLDER> is the folder where you have just downloaded the code in step 2

3. Install pytorch-ident in your python environment: run

.. code-block:: bash

   pip install .

from the command line, in the working folder LOCAL_FOLDER


Getting started
---------------
The best way to get started with pytorch-ident is to run the
`examples <https://github.com/forgi86/pytorch-ident/tree/master/examples>`_.

API Documentation
---------------------
.. toctree::
   :maxdepth: 2

   datasets
   statespace_dt
   statespace_ct
   dynonet
   metrics



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
