How to Run DomHMM
=================

This section explains how to configure and run DomHMM with several different options.

.. note::
    In the project's ``/example`` directory, you can find real-life examples of DomHMM.

Running DomHMM
--------------

DomHMM's main class is ``PropertyCalculation``. In a basic example, it is initialized as

.. code-block::

    model = domhmm.PropertyCalculation(universe_or_atomgroup=universe,
                                       leaflet_kwargs=leaflet_kwargs,
                                       membrane_select=membrane_select,
                                       leaflet_select="auto",
                                       heads=heads,
                                       sterol_heads=sterol_heads,
                                       sterol_tails=sterol_tails,
                                       tails=tails)

Then it can be run as

.. code-block::

    model.run(start=start_frame, end=end_frame, step=step)


Main Parameters
----------------

Let's dive into each parameter's details.

* In initialization process, ``universe_or_atomgroup`` parameter stands for MDAnalysis universe. It contains your simulation's trajectory and topology file. It can be created as

.. code-block::

    path2xtc = "YOUR_XTC_FILE.xtc"
    path2tpr = "YOUR_TPR_FILE.tpr"
    universe = mda.Universe(path2tpr, path2xtc)

* ``leaflet_kwargs`` parameter stands for MDAnalysis ``LeafletFinder`` function's arguments. It is used to determine each leaflet's lipids. ``leaflet_kwargs`` requires head groups of lipids but not sterols.

.. code-block::

    # An example where all lipids head group is PO4
    leaflet_kwargs={"select": "name PO4", "pbc": True}

* ``membrane_select`` argument is for atom group selection of universe. It is useful for simulations that contain non-membrane residues/molecules inside. If the universe contains only membrane elements, the parameter can be left in the default option which is ``all``

.. code-block::

    # An example where simulation contains DPPC and DIPC lipids, and CHOL sterol
    membrane_select = "resname DPPC DIPC CHOL"

* ``leaflet_select`` argument is a selection option for lipids which can be a list of atom groups, a list of string queries, or automatic via LeafletFinder.

.. code-block::

    # List of atom groups
    lower_leaflet = universe.select_atoms("lower leaflet lipids query")
    upper_leaflet = universe.select_atoms("upper leaflet lipids query")
    leaflet_select = [lower_leaflet, upper_leaflet]
    # List of query strings
    leaflet_select = ["lower leaflet lipids query", "upper leaflet lipids query"]
    # Leave leaflet detection to DomHMM via LeafletFinder
    leaflet_select = "auto"

* ``heads`` parameter requires lipids head groups. For atomistic simulations, the head molecules' center atom can be entered.

.. code-block::

    heads = {"DPPC": "PO4", "DIPC": "PO4"}

* ``sterol_heads`` parameter requires sterol head groups. For atomistic simulations, the head molecules' center atom can be entered.

.. code-block::

    # Martini Cholesterol example
    sterol_heads = {"CHOL": "ROH"}
    # Atomistic Cholesterol example
    sterol_heads = {"CHL1": "O3"}

* ``sterol_tails`` parameter requires sterol tail groups. It should be considered that each tail should be entered in the same order for each lipid.

.. code-block::

    # Martini Cholesterol example while ROH head as the first element and C1 start of the tail the second element
    sterol_tails = {"CHOL": ["ROH", "C1"]}
    # Atomistic Cholesterol example while O3 head as first element and C20 start of tail as second element
    sterol_tails = {"CHL1": ["O3", "C20"]}

* ``tails`` parameter requires lipids tail groups. It should be considered that each tail should be entered in the same order for each lipid.

.. code-block::

    # Example of tails in order of {"Lipid_1":[[Acyl_Chain_1],[Acyl_Chain_2]], "Lipid_2":[[Acyl_Chain_1],[Acyl_Chain_2]]}
    tails = {"DPPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "C2A", "C3A", "C4A"]],
                 "DIPC": [["C1B", "D2B", "D3B", "C4B"], ["C1A", "D2A", "D3A", "C4A"]]}


* For run option, you can have ``start``, ``stop`` and ``step`` options. These options arrange which frame to start or stop. You can also set the model to be trained for each *X* frame by setting ``step=X``.

.. code-block::

    # An example where DomHMM model training starts from the 5th frame and ends in the 1000th frame while taking each 5th step. The first three frames will be the 5th, 10th, and 15th frames.
    model.run(start=5, stop=1000, step=5)

.. warning::
    If detailed post-analysis will be conducted on results such as usage of ``Getis_Ord`` results, the input order of lipids and sterols should be in same order as in simulation. If simulation lipids are in order of ``DPPC, DIPC, CHOL`` with respect to residue IDs, keys of ``heads``, ``tails``, ``sterol_heads``, and ``sterol_tails`` should be in the same order just like in this example.

.. note::

    Since DomHMM uses the Gaussian Mixture Model and Gaussian-based Hidden Markov Model, it is suggested to not use too short or too long simulations. Short simulations may not create sensible results and long ones would take too much time to train the model. In our examples, we used simulations that contains around 2000 frames, and the model run is finished around 25-30 minutes.

Optional Parameters
-------------------

* ``asymmetric_membrane``

It needs to be enabled if leaflets are not symmetric. With this option, models are fitted by separated data for each leaflet.

* ``do_clustering``

Whether to perform the hierarchical clustering or not (Default is True).

* ``result_plot``

Plotting option for debugging. While enabled, DomHMM will print Hidden Markov model convergence, prediction results, Getis-Ord statistic results, and clustering results of three frames.

* ``save_plots``

Option for saving result plots in pdf format.

* ``verbose``

Verbose option for debugging. It shows which steps are done in the analysis.

* ``lipid_leaflet_rate``

The frame rate for checking lipids leaflet assignments via LeafletFinder. In the default option, it is equal to 0 which means leaflet assignment is only done at the beginning of the analysis.

* ``sterol_leaflet_rate``

The frame rate for checking sterols leaflet assignments via LeafletFinder. In the default option, it is equal to 1 which means sterols leaflet assignment will be calculated in every time frame to capture flip-flops.

* ``gmm_kwargs``

Parameter option for Gaussian Mixture Model training. An example of this is

.. code-block::

    gmm_kwargs = {"tol": 1E-4, "init_params": 'random_from_data', "verbose": 0,
                      "max_iter": 10000, "n_init": 20,
                      "warm_start": False, "covariance_type": "full"}

* ``hmm_kwargs``

Parameter option for Gaussian-based Hidden Markov Model training. An example of this is

.. code-block::

    hmm_kwargs = {"verbose": False, "tol": 1E-4, "n_iter": 1000,
                      "algorithm": "viterbi", "covariance_type": "full",
                      "init_params": "st", "params": "stmc"}

* ``trained_hmms``

Parameter option for reusing past DomHMM HMM models. If there are several analyses that will be conducted with slightly different membrane simulations or with different parameter options, the first analysis HMM model can be reusable with this parameter.

.. code-block::

    model.run()
    with open(f'hmm_model_dump.pickle', 'wb') as file:
        pickle.dump(model.results["HMM"], file)
    ...
    with open(f'hmm_model_dump.pickle', 'rb') as file:
        reuse_hmm_model = pickle.load(file)
    model_2 = domhmm.PropertyCalculation( ... ,
                                         trained_hmms=reuse_hmm_models)

* ``n_init_hmm``

Number of repeats for HMM model training. HMM models can be trained multiple times to achieve better performance.

* ``frac``

The fraction of box length in x and y outside the unit cell is considered for area per lipid calculation by Voronoi. It is an optimization process parameter that is set to 0.5 as the default.

* ``p_value``

Probability value that is used for z-score calculation. It is a determination percentage for domain identification with the Getis-Ord statistic. In default, it is set to 0.05 or %5.


* ``tmd_protein_list``

Transmembrane domain (TMD) protein list to include area per lipid calculation. TMD proteins take up space in the exoplasmic, cytoplasmic leaflets. Three backbone atoms of protein that are in close position to lipid head groups should be included in this parameter to increase the success of identification.

.. code-block::

    # Selecting three backbone atoms that are touching the exoplasmic leaflet
    upBB = uni.select_atoms('name BB')[0:3]
    # Selecting three backbone atoms that are touching the endoplasmic leaflet
    loBB = uni.select_atoms('name BB')[-3:]
    # List can be expended with multiple dictionary objects as in more than one TMD protein scenario.
    tmd_protein_list = [{"0": upBB, "1": loBB}]

We encourage you to check :doc:`tips` section which may contain useful information for your progress.