groups = {
    "aa_A": ["A"],
    "aa_C": ["C"],
    "aa_D": ["D"],
    "aa_E": ["E"],
    "aa_F": ["F"],
    "aa_G": ["G"],
    "aa_H": ["H"],
    "aa_I": ["I"],
    "aa_K": ["K"],
    "aa_L": ["L"],
    "aa_M": ["M"],
    "aa_N": ["N"],
    "aa_P": ["P"],
    "aa_Q": ["Q"],
    "aa_R": ["R"],
    "aa_S": ["S"],
    "aa_T": ["T"],
    "aa_U": ["U"],
    "aa_V": ["V"],
    "aa_W": ["W"],
    "aa_Y": ["Y"],
    #    "X": ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"],

    "grp_polar": ["N", "Q", "S", "T", "Y", "C"],
    "grp_nonpolar": ["A", "C", "F", "G", "I", "L", "M", "P", "V", "W"],
    "grp_charge": ["D", "E", "K", "H", "R"],
    "grp_positive": ["H", "K", "R"],
    "grp_negative": ["D", "E"],
    "grp_double": ["N", "Q"],
    "grp_sulfur-containing": ["C", "M"],
    "grp_aromatic": ["F", "W", "Y"],
    "grp_aliphatic": ["A", "G", "I", "V"],
    "grp_hydrophobic": ["A", "I", "L", "M", "V"],
    "grp_structural_breaker": ["P", "G"],
    "grp_small": ["G", "A"],
    "grp_coord_metal_ions": ["H", "C"],
    "grp_phosphorylation": ["T", "S", "Y"],
    "grp_methylation": ["R", "K"],
    "grp_neutralH": ["A", "C", "F", "G", "H", "I", "L", "M", "N", "P", "Q", "S", "T", "V", "W", "Y"],
    "grp_neutral": ["A", "C", "F", "G", "I", "L", "M", "N", "P", "Q", "S", "T", "V", "W", "Y"],
}

protein_letters_1to3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "U": "TMP",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

a2gs = {}
group_list = []
for g, aas in groups.items():
    for a in aas:
        if g.startswith('aa_'):
            a2gs.setdefault(protein_letters_1to3[a], [])
        else:
            if g not in group_list:
                group_list.append(g)
            a2gs.setdefault(protein_letters_1to3[a], []).append(g)
