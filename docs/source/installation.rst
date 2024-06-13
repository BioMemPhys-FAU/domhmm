Installation
=============
Installation can be done in two different ways

.. note::
    DomHMM is planning to be published also in PyPI. Which means in future, it can be directly install via pip and can be usable without any further actions

GitHub Repository
------------------

Clone DomHMM's repository and change directory to project directory

.. code-block::

    git clone https://github.com/m-a-r-i-u-s/domhmm
    cd domhmm

Environment Creation
---------------------
If you don't have any Python environment, you can create a conda environment by

.. code-block:: console

    conda create --name domhmm
    conda activate domhmm

.. tip::

    Create command creates a conda environment named domhmm and activate command activates it. Whenever you close your terminal, you need to use activate command again.

After conda environment creation, you can install dependencies via

.. code-block:: console

    pip install -e .

Installation via pip
---------------------

For direct installation, you can directly use pip in project directory.

.. code-block:: console

   pip install -e .


Check out the :doc:`how-to-use` section as next step.
