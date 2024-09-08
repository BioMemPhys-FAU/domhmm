import os

import MDAnalysis as mda

import domhmm

if __name__ == "__main__":
    # MDAnalysis universe for membrane simulation
    data_dir = os.path.dirname(__file__)
    path2xtc = os.path.join(data_dir, "md_center_mol.xtc")
    path2tpr = os.path.join(data_dir, "mem.tpr")
    uni = mda.Universe(path2tpr, path2xtc)

    # Selecting three backbone atoms that is touching to upper leaflet
    upBB = uni.select_atoms('name BB')[0:3]
    # Selecting three backbone atoms that is touching to lower leaflet
    loBB = uni.select_atoms('name BB')[-3:]

    tmd_protein_list = [{"0": upBB, "1": loBB}]
    membrane_select = "resname PUPC POPC CHOL"
    heads = {"PUPC": "PO4", "POPC": "PO4"}
    tails = {"POPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "D2A", "C3A", "C4A"]],
             "PUPC": [["C1B", "C2B", "C3B", "C4B"], ["D1A", "D2A", "D3A", "D4A", "D5A"]]}
    sterol_heads = {"CHOL": "ROH"}
    sterol_tails = {"CHOL": ["ROH", "C1"]}

    # leaflet_kwargs should contain all head group atoms/molecules of lipids for LeafletFinder function
    model = domhmm.PropertyCalculation(universe_or_atomgroup=uni,
                                       leaflet_kwargs={"select": "name P*", "pbc": True},
                                       membrane_select=membrane_select,
                                       leaflet_select="auto",
                                       heads=heads,
                                       sterol_heads=sterol_heads,
                                       sterol_tails=sterol_tails,
                                       tails=tails,
                                       result_plots=True,
                                       tmd_protein_list=tmd_protein_list,
                                       )

    # Run option can be updated by parameters such as start=0, stop=100, step=5
    model.run(start=0, stop=100)
