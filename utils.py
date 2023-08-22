import os

import mfgb.features

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops, AllChem
import numpy as np
import openbabel as ob

kg_dict = {}

with open('triples-mgbert.txt', 'r', encoding='utf-8') as opf:
    lines = opf.readlines()
    for line in lines:
        line = line.strip().split(',')
        if line[0] in kg_dict:
            kg_dict[line[0]].append((line[1], line[2]))
        else:
            kg_dict[line[0]] = []
            kg_dict[line[0]].append((line[1], line[2]))
opf.close()


def mol2alt_sentence_rad(mol, radius):
    """Same as mol2sentence() expect it only returns the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius

    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)

def mol2alt_sentence_r2(mol, radius):
    """Same as mol2sentence() expect it only returns the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius

    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        # for r in radii:  # iterate over radii
        if dict_atoms[atom][2] != None:
            identifiers_alt.append(dict_atoms[atom][2])
        else:
            identifiers_alt.append(dict_atoms[atom][0])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


def mol2alt_sentence_r1(mol, radius):
    """Same as mol2sentence() expect it only returns the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius

    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        # for r in radii:  # iterate over radii
        if dict_atoms[atom][1] != None:
            identifiers_alt.append(dict_atoms[atom][1])
        else:
            identifiers_alt.append(dict_atoms[atom][0])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)



def obsmitosmile(smi):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = ob.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile


def smiles2adjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    adjoin_matrix = np.eye(num_atoms)
    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0
    return atoms_list, adjoin_matrix


def glgsmiles2adjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())


    num_bonds = mol.GetNumBonds()


    bonds_list = []
    nei_bond = []
    for i in range(num_bonds):
        nei_bond.append((mol.GetBondWithIdx(i).GetBondType().name,
                         mol.GetBondWithIdx(i).GetBeginAtomIdx(),
                         mol.GetBondWithIdx(i).GetEndAtomIdx()))
        bonds_list.append(mol.GetBondWithIdx(i).GetBondType().name)

    if len(atoms_list) > len(bonds_list):
        adjoin_num = len(atoms_list)
    else:
        adjoin_num = len(bonds_list)

    atom_adjoin_matrix = np.eye(adjoin_num)
    # Add edges
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        atom_adjoin_matrix[u, v] = 1.0
        atom_adjoin_matrix[v, u] = 1.0

    print(atoms_list)
    print(atom_adjoin_matrix)

    bond_adjoin_matrix = np.eye(adjoin_num)
    for i in range(len(nei_bond)):
        iatom_set = set()
        iatom_set.add(nei_bond[i][1])
        iatom_set.add(nei_bond[i][2])
        for j in range(len(nei_bond)):
            jatom_set = set()
            jatom_set.add(nei_bond[j][1])
            jatom_set.add(nei_bond[j][2])
            if iatom_set.isdisjoint(jatom_set) == False:
                bond_adjoin_matrix[i][j] = 1
                bond_adjoin_matrix[j][i] = 1
    print(bonds_list)
    print(bond_adjoin_matrix)

    return atoms_list, atom_adjoin_matrix, bonds_list, bond_adjoin_matrix


def smiles2kgadjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    rel_list = []
    ent_list = []
    for atom in atoms_list:
        if atom in kg_dict:
            triples = kg_dict[atom]
            for item in triples:
                rel_list.append(item[0])
                ent_list.append(item[1])

    rel_set = set(rel_list)
    ent_set = set(ent_list)
    rel_set_list = list(rel_set)
    ent_set_list = list(ent_set)
    rel_set_list.sort()
    ent_set_list.sort()

    num_rels = len(rel_set_list)
    num_ents = len(ent_set_list)


    atoms_rel_ent_list = []
    atoms_rel_ent_list.extend(atoms_list)
    atoms_rel_ent_list.extend(rel_set_list)
    atoms_rel_ent_list.extend(ent_set_list)

    idx_dict = dict()
    for i in range(len(rel_set_list)):
        idx_dict[rel_set_list[i]] = len(atoms_list) + i

    for i in range(len(ent_set_list)):
        idx_dict[ent_set_list[i]] = len(atoms_list) + len(rel_set_list) + i

    # for k,v in idx_dict.items():
    #     if k != atoms_rel_ent_list[v]:
    #         print(k)
    #         print(atoms_rel_ent_list[v])
    #         print("Error")

    adjoin_matrix = np.eye(num_atoms + num_rels + num_ents)
    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0

    for i in range(len(atoms_list)):
        atom = atoms_list[i]
        if atom in kg_dict:
            triples = kg_dict[atom]
            for item in triples:
                rel_idx = idx_dict[item[0]]
                ent_idx = idx_dict[item[1]]
                adjoin_matrix[i, rel_idx] = 1.0
                adjoin_matrix[rel_idx, i] = 1.0
                adjoin_matrix[rel_idx, ent_idx] = 1.0
                adjoin_matrix[ent_idx, rel_idx] = 1.0

    # print(len(atoms_rel_ent_list))
    # print(len(adjoin_matrix[0]))
    # print(adjoin_matrix)
    # for i in range(len(adjoin_matrix[10])):
    #     if adjoin_matrix[10][i] != 0 and i >= 10:
    #         print(atoms_rel_ent_list[i])
    return atoms_rel_ent_list, adjoin_matrix

def smiles2kgteadjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    # rel_list = []
    ent_list = []
    for atom in atoms_list:
        if atom in kg_dict:
            triples = kg_dict[atom]
            for item in triples:
                # rel_list.append(item[0])
                ent_list.append(item[1])

    # rel_set = set(rel_list)
    ent_set = set(ent_list)
    # rel_set_list = list(rel_set)
    ent_set_list = list(ent_set)
    # rel_set_list.sort()
    ent_set_list.sort()

    # num_rels = len(rel_set_list)
    num_ents = len(ent_set_list)

    # print(atoms_list)
    # print(rel_set_list)
    # print(ent_set_list)

    atoms_ent_list = []
    atoms_ent_list.extend(atoms_list)
    # atoms_rel_ent_list.extend(rel_set_list)
    atoms_ent_list.extend(ent_set_list)

    # type encoding
    # type_list = []
    # atom_type_list = [1] * len(atoms_list)
    # ent_type_list = [2] * len(ent_set_list)
    # type_list.extend(atom_type_list)
    # type_list.extend(ent_type_list)

    # print(type_list)

    idx_dict = dict()
    # for i in range(len(rel_set_list)):
    #     idx_dict[rel_set_list[i]] = len(atoms_list) + i

    for i in range(len(ent_set_list)):
        idx_dict[ent_set_list[i]] = len(atoms_list) + i

    # for k,v in idx_dict.items():
    #     if k != atoms_rel_ent_list[v]:
    #         print(k)
    #         print(atoms_rel_ent_list[v])
    #         print("Error")

    adjoin_matrix = np.eye(num_atoms + num_ents)
    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0

    for i in range(len(atoms_list)):
        atom = atoms_list[i]
        if atom in kg_dict:
            triples = kg_dict[atom]
            for item in triples:
                # rel_idx = idx_dict[item[0]]
                ent_idx = idx_dict[item[1]]
                # adjoin_matrix[i, rel_idx] = 1.0
                # adjoin_matrix[rel_idx, i] = 1.0
                adjoin_matrix[i, ent_idx] = 1.0
                adjoin_matrix[ent_idx, i] = 1.0

    print(atoms_ent_list)

    print(adjoin_matrix)

    # print(len(atoms_ent_list))
    # print(len(adjoin_matrix[0]))
    # print(adjoin_matrix)
    for i in range(len(adjoin_matrix)):
        for j in range(len(adjoin_matrix[i])):
            if adjoin_matrix[i][j] == 1 and i != j:
                print([atoms_ent_list[i], atoms_ent_list[j]])
    return atoms_ent_list, adjoin_matrix, num_atoms

def smiles2mfgbadjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    # atoms_list = []
    # for i in range(num_atoms):
    #     atom = mol.GetAtomWithIdx(i)
    #     atoms_list.append(atom.GetSymbol())
    # print(atoms_list)
    ident_list = mol2alt_sentence_rad(mol, 0)

    adjoin_matrix = np.eye(num_atoms)
    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0

    # print(adjoin_matrix)

    # for i in range(len(adjoin_matrix)):
    #     for j in range(len(adjoin_matrix[i])):
    #         if adjoin_matrix[i][j] == 1 and i != j:
    #             print([ident_list[i], ident_list[j]])

    return ident_list, adjoin_matrix

def smiles2mfgbadjoin_r1(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())
    # print(atoms_list)
    # print(len(atoms_list))
    ident_list = mol2alt_sentence_r1(mol, 1)

    if len(ident_list) != num_atoms:
        assert smiles + ' mol2alt_sentence_r1 anomalies'
    # print(ident_list)
    # print(len(ident_list))
    # ident_list_new = mol2alt_sentence_rad(mol, 0)
    # print(ident_list_new)
    # print(len(ident_list_new))

    adjoin_matrix = np.eye(num_atoms)
    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0

    # print(adjoin_matrix.shape)
    # print(adjoin_matrix)

    # print(adjoin_matrix)

    # for i in range(len(adjoin_matrix)):
    #     for j in range(len(adjoin_matrix[i])):
    #         if adjoin_matrix[i][j] == 1 and i != j:
    #             print([ident_list[i], ident_list[j]])

    return ident_list, adjoin_matrix

def smiles2mfgbadjoin_r2(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())
    # print(atoms_list)
    # print(len(atoms_list))
    ident_list = mol2alt_sentence_r2(mol, 2)

    if len(ident_list) != num_atoms:
        assert smiles + ' mol2alt_sentence_r2 anomalies'
    # print(ident_list)
    # print(len(ident_list))
    # ident_list_new = mol2alt_sentence_rad(mol, 0)
    # print(ident_list_new)
    # print(len(ident_list_new))

    adjoin_matrix = np.eye(num_atoms)
    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0

    # print(adjoin_matrix.shape)
    # print(adjoin_matrix)

    # print(adjoin_matrix)

    # for i in range(len(adjoin_matrix)):
    #     for j in range(len(adjoin_matrix[i])):
    #         if adjoin_matrix[i][j] == 1 and i != j:
    #             print([ident_list[i], ident_list[j]])

    return ident_list, adjoin_matrix

def atombondsmiles2adjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []

    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    atom_type = [1] * len(atoms_list)

    num_bonds = mol.GetNumBonds()
    bond_list = []
    bond_name_list = []
    for i in range(num_bonds):
        bond_list.append((mol.GetBondWithIdx(i).GetBondType().name, mol.GetBondWithIdx(i).GetBeginAtomIdx(),
                         mol.GetBondWithIdx(i).GetEndAtomIdx()))
        bond_name_list.append(mol.GetBondWithIdx(i).GetBondType().name)

    bond_type = [2] * len(bond_name_list)

    # print(bond_list)
    # print(bond_name_list)
    adjoin_matrix = np.eye(num_atoms + num_bonds)
    atom_bond_list = []
    atom_bond_list.extend(atoms_list)
    atom_bond_list.extend(bond_name_list)

    type_list = []
    type_list.extend(atom_type)
    type_list.extend(bond_type)

    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        b = i + num_atoms
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, b] = 1.0
        adjoin_matrix[b, u] = 1.0
        adjoin_matrix[v, b] = 1.0
        adjoin_matrix[b, v] = 1.0

    # for i in range(len(adjoin_matrix)):
    #     for j in range(len(adjoin_matrix[i])):
    #         if adjoin_matrix[i][j] == 1 and i != j:
    #             print([atom_bond_list[i], atom_bond_list[j]])
    #
    # print(adjoin_matrix)
    # print(atom_bond_list)
    # print(type_list)

    return atom_bond_list, type_list, adjoin_matrix

if __name__ == "__main__":
   smiles2mfgbadjoin_r1("C1=CC(=CC=C1SC(P(=O)(O)[O-])P(=O)(O)[O-])Cl.[Na+].[Na+]", explicit_hydrogens=False, canonical_atom_order=False)
