import os
import pickle

import MDAnalysis as mda

import domhmm

if __name__ == "__main__":
    # MDAnalysis universe for membrane simulation
    # Test data is being used in this example
    data_dir = os.path.dirname(__file__)
    path2xtc = os.path.join(data_dir, "md_center_mol_last2mus.xtc")
    path2tpr = os.path.join(data_dir, "mem.tpr")
    universe = mda.Universe(path2tpr, path2xtc)

    # membrane_select can be "all" if there aren't any other molecules rather then membrane lipids and sterols
    membrane_select = "resname DPPC DIPC CHOL"

    # For martini simulations heads are defined as head molecule
    heads = {"DPPC": "PO4",
             "DIPC": "PO4"}

    # Order of the tails should be same for each lipids.
    # If first lipid's array is first acyl chain then other lipids first array should be also first acyl chain
    tails = {"DPPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "C2A", "C3A", "C4A"]],
             "DIPC": [["C1B", "D2B", "D3B", "C4B"], ["C1A", "D2A", "D3A", "C4A"]]}

    # In Sterols, there is one dimension array which first element is represents head part and second element
    # presents beginning part of sterol's chain
    sterols = {"CHOL": ["ROH", "C1"]}
    sterol_heads = {"CHOL": "ROH"}
    sterol_tails = {"CHOL": ["ROH", "C1"]}

    model = domhmm.PropertyCalculation(universe_or_atomgroup=universe,
                                       membrane_select=membrane_select,
                                       leaflet_select="auto",
                                       heads=heads,
                                       sterol_heads=sterol_heads,
                                       sterol_tails=sterol_tails,
                                       tails=tails,
                                       result_plots=True,
                                       save_plots=True,
                                       parallel_clustering=True,
                                       verbose=True,
                                       )

    # run option can be updated by parameters such as start=0, stop=100, step=5
    model.run()

    # For further work, user can use training data which contains area per lipid and Scc order parameters
    data = model.results["train_data_per_type"]

    # Residue ids in the same order with training data rows
    dppc_res_ids = data["DPPC"][0]

    # Area per lipid and Scc order parameter for each residues each frame in order [[apl_1, scc_1_1, scc_1_2], ...]
    # For Sterols, there is only one
    dppc_parameters = data["DPPC"][1]

    # Model's itself or required result sections can be save via pickle
    with open('model_dump.pickle', 'wb') as file:
        pickle.dump(model, file)

    # Model can be load again with pickle
    with open('model_dump.pickle', 'rb') as file:
        loaded_module = pickle.load(file)
