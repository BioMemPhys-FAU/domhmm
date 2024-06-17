How to Run DomHMM
=================

This section is about how to use DomHMM and how to elaborate on results.

.. note::
    In project's ``/example`` directory, you can find real life usage of DomHMM.

Running DomHMM
--------------

DomHMM's main class is ``PropertyCalculation``. In a basic example it is initialized as

.. code-block::

    model = domhmm.PropertyCalculation(universe_or_atomgroup=universe,
                                       leaflet_kwargs=leaflet_kwargs,
                                       membrane_select=membrane_select,
                                       heads=heads,
                                       sterols=sterols,
                                       tails=tails)

Then it can be run as

.. code-block::

    model.run(start=start, end=end, step=step)


Main Parameters
----------------

Let's dive into each parameter's details.

* In initialization process, ``universe_or_atomgroup`` parameter stands for MDAnalysis universe. It contains your simulation's trajectory and tpr file. It can be created as

.. code-block::

    path2xtc = "YOUR_XTC_FILE.xtc"
    path2tpr = "YOUR_TPR_FILE.tpr"
    universe = mda.Universe(path2tpr, path2xtc)

* ``leaflet_kwargs`` parameter stands for MDAnalysis ``LeafletFinder`` function's arguments. It is used to determine each leaflets residues. ``leaflet_kwargs`` requires head groups of lipids but not sterols.

.. code-block::

    # An example where all lipids head group is PO4
    leaflet_kwargs={"select": "name PO4", "pbc": True}

* ``membrane_select`` argument is for atom group selection of universe. It is useful for simulations that are contain non-membrane residues/molecules inside. If universe contains only membrane elements parameter can be leave in default option which is ``all``

.. code-block::

    # An example where simulation contains DPPC and DIPC lipids, and CHOL sterol
    membrane_select = "resname DPPC DIPC CHOL"

* ``heads`` parameter requires lipids head groups. For atomistic simulations, head molecules' center atom can be entered.

.. code-block::

    heads = {"DPPC": "PO4", "DIPC": "PO4"}

* ``tails`` parameter requires lipids tail groups. It should be considered that each tail should be entered in same order for each lipids.

.. code-block::

    # Example of tails in order of {"Lipid_1":[[Acyl_Chain_1],[Acyl_Chain_2]], "Lipid_2":[[Acyl_Chain_1],[Acyl_Chain_2]]}
    tails = {"DPPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "C2A", "C3A", "C4A"]],
                 "DIPC": [["C1B", "D2B", "D3B", "C4B"], ["C1A", "D2A", "D3A", "C4A"]]}

* ``sterols`` parameter requires head and tail group of each sterol. It is important to put head as first element and tails' starting point as second element.

.. code-block::

    # Martini Cholestrol example while ROH as head group and C1 as start of tail
    sterols = {"CHOL": ["ROH", "C1"]}
    # Atomistic Cholestrol example while O3 as head group and C20 as start of tail
    sterols = {"CHL1": ["O3", "C20"]}


* For run option, you can have ``start``, ``stop`` and ``step`` options. This options arrange which frame to start, stop. You can also set model to be trained for each *X* frame by setting ``step=X``.

.. code-block::

    # An example where DomHMM model training starts from 5th frame and ends in 1000th frame while taking each 5th step. First three frames will be 5th, 10th and 15th frames.
    model.run(start=5, stop=1000, step=5)

.. warning::

    Since DomHMM uses Gaussian Mixture Model and Gaussian-based Hidden Markov Model, it is suggested to not use too short or too long simulations. Short simulations may not create a sensible results and long one would be take too much time to train model. In our examples, we used simulations that contains around 2000 frames and model run is finished around 25-30 minutes.

Optional Parameters
-------------------

* ``frac``

Fraction of box length in x and y outside the unit cell considered for area per lipid calculation by Voronoi. It is an optimization process parameter which is set to 0.5 as default.

* ``p_value``

Probability value that is used for z-score calculation. It is a determination percentage for domain identification with getis-ord statistic. In default, it is set to 0.05 or %5.

* ``verbose``

Verbose option for debugging. Although, DomHMM doesn't print middle values, it shows which steps are done and shows middle step plots which may give clues about succession of model.


Future Optional Parameters
--------------------------

.. note::
    This parameters will be added to DomHMM in future before release


* ``gmm_kwargs``

Parameter option for Gaussian Mixture Model training. An example of it is

.. code-block::

    gmm_kwargs = {"tol": 1E-4, "init_params": 'k-means++', "verbose": 0,
                      "max_iter": 10000, "n_init": 20,
                      "warm_start": False, "covariance_type": "full"}

* ``hmm_kwargs``

Parameter option for Gaussian-based Hidden Markov Model training. An example of it is

.. code-block::

    hmm_kwargs = {"verbose": False, "tol": 1E-4, "n_iter": 1000,
                      "algorithm": "viterbi", "covariance_type": "full",
                      "init_params": "st", "params": "stmc"}


We encourage to check :doc:`tips` section that may contain useful information for your progress.