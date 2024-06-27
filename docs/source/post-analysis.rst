Results and Post-Analysis
==========================

After running of DomHMM, results are achievable via assigned variable which in this document named ``model``. Besides clustering results of ordered and disorder domains, training data that is used for Hidden Markov Model is also available which contains area per lipid calculation and Scc order parameters calculations for each lipid and sterol.

Domain Cluster Results
-----------------------
``Clustering`` is a Python dictionary which contains each frames residue indexes that are assigned to Lo ordered domains.

.. code-block::

    clusters = model.results["Clustering"]
    first_frame_clusters = clusters[0]
    first_frame_number_of_clusters = len(clusters[0])
    fırst_frame_first_cluster = first_frame_clusters[0]

.. note::

    Clustering result dictionary is in format such as ``{frame_number : [[Cluster 1 Residue Indexes], [Cluster 2 Residue Indexes]], frame_number_2: [[Cluster 1 Residue Indexes], [Cluster 2 Residue Indexes]], ...]}``


Training Data (Area per lipid and order parameters)
---------------------------------------------------

If required for post analysis, user can access area per lipid and order parameters calculations of each lipid. This data is kept objects result data which can be accessed via ``model.results["train_data_per_type"]``.

``train_data_per_type`` is a Python dictionary which contains lipid and sterol names are keys and three dimension arrays as values. In this three dimension array, each dimension contains residue ids, second dimension contains parameters and third dimension contains each frame's residue leaflet assignments.
Be aware that both second and third arrays are in same order of residue ids from first array.

Here is an example of it.

.. code-block::

    data = model.results["train_data_per_type"]
    dppc_res_ids = data["DPPC"][0]
    dppc_parameters = data["DPPC"][1]
    dppc_leaflet_assignment = data["DPPC"][2]
    print(f"Residue Id List: \n {res_ids}")
    print(f"Parameters: \n {parameters}")
    print(f"Leaflet Assignments: \n {leaflet_assignment}")

.. note::

    Each arrays are in ``numpy.array`` format.

.. note::
    Parameters array (second array) is keep in order of ``[[apl_1, scc_1_1, scc_1_2],[apl_2, scc_2_1, scc_2_2], ...]``. (apl = Area per Lipid, scc__x= Scc Order Parameter of tail x )

.. note::
    Leaflet assignment array (third array) is consists of 0s and 1s where 0 means upper leaflet and 1 means lower leaflet. Rows are represents residues which are in some order with residue ids from first array and columns are represents frames.

.. note::
    Names of lipids and sterols are same names that user gave in tails and heads parameters.


Result Saving
---------------
User can save and reload model's itself or required data via `pickle`_.

.. code-block::

    # Model's itself or required result sections can be save via pickle
    with open('DomHMM_model.pickle', 'wb') as file:
        pickle.dump(model, file)

    # Model can be reload again with pickle
    with open('DomHMM_model.pickle', 'rb') as file:
        loaded_module = pickle.load(file)



.. _pickle: https://www.mdanalysis.org/pages/mdakits/