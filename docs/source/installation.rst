Installation
=============
Installation can be done in two different ways


Environment Creation
---------------------
If you don't have any Python environment, you can create a conda environment by

.. code-block:: console

    conda create --name domhmm
    conda activate domhmm

.. tip::

    ``conda create`` command creates a conda environment named *domhmm* and ``conda activate`` command starts environment in your terminal. Whenever you close your terminal, you need to use activate command again to restart environment.


Installation with PyPI
-----------------------

For installation, you can directly use pip in project directory.

.. code-block:: console

   pip install domhmm


Installation for Development
------------------------------

This type of installation can be use when source code will be change for special usage or contribution will be done to DomHMM.

Clone DomHMM's repository and change directory to project directory

.. code-block::

    git clone https://github.com/BioMemPhys-FAU/domhmm
    cd domhmm

Install dependencies and DomHMM's current version via pip command

.. code-block:: console

   pip install -e .


Check out the :doc:`how-to-run` section as next step.
