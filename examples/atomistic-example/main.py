import os

import MDAnalysis as mda

import domhmm

if __name__ == "__main__":
    # MDAnalysis universe for membrane simulation
    data_dir = os.path.dirname(__file__)
    path2xtc = os.path.join(data_dir, "PM_176sern_600ns_10fr.xtc")
    path2tpr = os.path.join(data_dir, "PM_176sern_membsernonly.tpr")
    uni = mda.Universe(path2tpr, path2xtc)
    # membrane_select can be "all" if there aren't any other molecules rather then membrane lipids and sterols
    membrane_select = "resname PSM POPC CHL1"
    # For atomistic simulations heads are defined as main atom in head molecule
    heads = {"PSM": "P", "POPC": "P"}
    # Order of the tails should be same for each lipids.
    # If first lipid's array is first acyl chain then other lipids first array should be also first acyl chain
    tails = {"POPC": [['C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C310', 'C311', 'C312', 'C313', 'C314', 'C315', 'C316'],
                      ['C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C210', 'C211', 'C212', 'C213', 'C214', 'C215', 'C216', 'C217', 'C218']],
             "PSM": [['C2F', 'C3F', 'C4F', 'C5F', 'C6F', 'C7F', 'C8F', 'C9F', 'C10F', 'C11F', 'C12F', 'C13F', 'C14F', 'C15F', 'C16F'],
                     ['C4S', 'C5S', 'C6S', 'C7S', 'C8S', 'C9S', 'C10S', 'C11S', 'C12S', 'C13S', 'C14S', 'C15S', 'C16S', 'C17S', 'C18S']]}
    # In Sterols, there is one dimension array which first element is represents head part and second element
    # presents beggining part of sterol's chain
    sterols = {"CHL1": ["O3", "C20"]}
    # leaflet_kwargs should contain all head group atoms/molecules of lipids for LeafletFinder function
    model = domhmm.PropertyCalculation(universe_or_atomgroup=uni,
                                       leaflet_kwargs={"select": "name P*", "pbc": True},
                                       membrane_select=membrane_select,
                                       heads=heads,
                                       sterols=sterols,
                                       tails=tails,
                                       verbose=True)
    # run option can be updated by parameters such as start=0, stop=100, step=5
    model.run()
    # TODO Result part for post analysis