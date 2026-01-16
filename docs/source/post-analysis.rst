Results and Post-Analysis
==========================

After running DomHMM, results are achievable via an assigned variable which in this document is named ``model``. Besides clustering results of ordered and disorder domains, training data that is used for the Hidden Markov model is also available which contains area per lipid calculation and Scc order parameters calculations for each lipid and sterol.

Domain Cluster Results
-----------------------
``Clustering`` is a Python dictionary that contains each frame residue index that is assigned to lipid-ordered domains.

``Clustering`` is a dictionary with two keys ``"0"`` as representing upper leaflet and ``"1"`` as representing lower leaflet.

.. code-block::

    clusters = model.results["Clustering"]
    upper_leaflet_clusters = clusters["0"]
    lower_leaflet_clusters = clusters["1"]
    upper_first_frame_clusters = upper_leaflet_clusters[0]
    upper_first_frame_number_of_clusters = len(upper_leaflet_clusters[0])
    upper_first_frame_first_cluster = upper_first_frame_clusters[0]

.. note::

    Clustering result dictionary is in format such as ``{"0": {frame_number : [[Cluster 1 Residue Indexes], [Cluster 2 Residue Indexes]], frame_number_2: [[Cluster 1 Residue Indexes], [Cluster 2 Residue Indexes]], ...]}, "1": ...}``


Training Data (Area per lipid and order parameters)
---------------------------------------------------

If required for post-analysis, the user can access the area per lipid and order parameter calculations of each lipid. This data is kept objects result in data which can be accessed via ``model.results["train_data_per_type"]``.

``train_data_per_type`` is a Python dictionary that contains lipid names as keys and three rowed arrays as values. The first row contains residue IDs, the second training data, and the third each frame's residue leaflet assignments.
Be aware that both the second and third arrays are in the same order of residue IDs from the first array.

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

    Each array is in ``numpy.array`` format.

.. note::
    Parameters array (second array) is kept in order of ``[[apl_1, scc_1_1, scc_1_2],[apl_2, scc_2_1, scc_2_2], ...]``. (apl = Area per Lipid, scc__x= Scc Order Parameter of tail x )

.. note::
    The leaflet assignment array (third array) consists of 0s and 1s where 0 means exoplasmic leaflet and 1 means endoplasmic leaflet. Rows represent residues which are in some order with residue IDs from the first array and columns represent frames.

.. note::
    Names of lipids and sterols are the same names that users gave in tails and heads parameters.


Result Saving
---------------
Users can save and reload the model itself or required data via `pickle`_.

.. code-block::

    # Model itself or result section can be saved via pickle
    with open('DomHMM_model.pickle', 'wb') as file:
        pickle.dump(model, file)

    # Model can be reload again with pickle
    with open('DomHMM_model.pickle', 'rb') as file:
        loaded_module = pickle.load(file)


.. note::
    When loading the full model, the MDAnalysis universe will load the trajectory and topology file from the same directory that was given in the analysis run. Therefore, full-model saving may not be loaded from outside of analysis directory.

.. _pickle: https://www.mdanalysis.org/pages/mdakits/