import os

import MDAnalysis as mda

import domhmm

if __name__ == "__main__":
    # MDAnalysis universe for membrane simulation
    # Test data is being used in this example
    data_dir = '../../domhmm/tests/data'
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
    # presents beggining part of sterol's chain
    sterols = {"CHOL": ["ROH", "C1"]}
    # leaflet_kwargs should contain all head group molecules of lipids for LeafletFinder function
    model = domhmm.PropertyCalculation(universe_or_atomgroup=universe,
                                      leaflet_kwargs={"select": "name PO4", "pbc": True},
                                      membrane_select=membrane_select,
                                      heads=heads,
                                      sterols=sterols,
                                      tails=tails)
    # run option can be updated by parameters such as start=0, stop=100, step=5
    model.run(start=0, stop=100)
    # TODO Result part for post analysis