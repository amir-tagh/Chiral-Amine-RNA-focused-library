import csv
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from multiprocessing import Pool, cpu_count


def has_free_amine(mol):
    """
    Check if a molecule contains an aliphatic primary amine.
    """
    aliphatic_primary_amine_pattern = Chem.MolFromSmarts('[N;H2;!$(NC=O);!a]')
    return mol.HasSubstructMatch(aliphatic_primary_amine_pattern)


def is_chiral(mol):
    """Check if the molecule has known chiral centers."""
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    return bool(chiral_centers)


def exclude_aromatic_nh(mol):
    """
    Exclude molecules with an NH group as part of the aromatic system.
    """
    aromatic_nh_pattern = Chem.MolFromSmarts('[nH]')
    return not mol.HasSubstructMatch(aromatic_nh_pattern)


def has_heterocycle(mol):
    """
    Check if the molecule contains at least one heterocycle.
    
    A heterocycle is a ring structure with at least one heteroatom (N, O, S).
    """
    # SMARTS pattern for heterocycles
    heterocycle_pattern = Chem.MolFromSmarts('[r5,r6;!#6]')
    return mol.HasSubstructMatch(heterocycle_pattern)



def exclude_acids_and_hydroxy(mol):
    """
    Exclude molecules with acid functional groups or hydroxy groups.
    Includes carboxylic acids, sulfonic acids, phosphoric acids, and hydroxy groups.
    """
    exclusion_patterns = [
        Chem.MolFromSmarts('C(=O)[OH]'),  # Carboxylic acid
        Chem.MolFromSmarts('S(=O)(=O)[OH]'),  # Sulfonic acid
        Chem.MolFromSmarts('P(=O)(O)[OH]'),   # Phosphoric acid
        Chem.MolFromSmarts('[OH]')  # Hydroxy group
    ]
    for pattern in exclusion_patterns:
        if mol.HasSubstructMatch(pattern):
            return False
    return True


def filter_by_molecular_weight_range(mol, mw_min, mw_max):
    """
    Check if the molecule's molecular weight is within the specified range.
    """
    mol_weight = Descriptors.MolWt(mol)
    return mw_min <= mol_weight <= mw_max


def filter_by_qed(mol, qed_cutoff):
    """Check if the molecule's QED value is above the cutoff."""
    qed_value = QED.qed(mol)
    return qed_value >= qed_cutoff


def process_smiles(smiles, mw_min=None, mw_max=None, qed_cutoff=None):
    """
    Process a single SMILES string to check conditions.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        if (is_chiral(mol) and has_free_amine(mol) and exclude_aromatic_nh(mol) and has_heterocycle(mol) and exclude_acids_and_hydroxy(mol)):
            if (mw_min is None or mw_max is None or filter_by_molecular_weight_range(mol, mw_min, mw_max)) and \
               (qed_cutoff is None or filter_by_qed(mol, qed_cutoff)):
                return smiles, mol
    return None


def search_smiles(smiles_list, mw_min=None, mw_max=None, qed_cutoff=None):
    """Filter SMILES strings based on multiple conditions."""
    selected_smiles = []
    selected_mols = []

    # Use all available CPUs for multiprocessing
    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_smiles, [(smiles, mw_min, mw_max, qed_cutoff) for smiles in smiles_list])

    # Filter results and append to the final lists
    for result in results:
        if result:
            smiles, mol = result
            selected_smiles.append(smiles)
            selected_mols.append(mol)
            print(f"Chiral centers for {smiles}: {Chem.FindMolChiralCenters(mol, includeUnassigned=False)}")
            print(f"Free amine found in {smiles}")
            if mw_min is not None and mw_max is not None:
                print(f"Molecular weight of {smiles}: {Descriptors.MolWt(mol)}")
            if qed_cutoff:
                print(f"QED value of {smiles}: {QED.qed(mol)}")

    return selected_smiles, selected_mols


def read_smiles_from_csv(input_file):
    """Read SMILES strings from a CSV file."""
    smiles_list = []
    with open(input_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                smiles_list.append(row[0])
    return smiles_list


def write_smiles_to_csv(output_file, smiles_list):
    """Write selected SMILES strings to a CSV file."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for smiles in smiles_list:
            writer.writerow([smiles])


def write_smiles_to_sdf(output_file, mols):
    """Write selected molecules to an SDF file."""
    writer = Chem.SDWriter(output_file)
    for mol in mols:
        if mol:
            # Generate 3D coordinates if not present
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
            writer.write(mol)
    writer.close()


def print_help():
    """Print help message."""
    help_message = """
Usage: python select_free_amines.py [OPTIONS]

Options:
  -i, --input FILE       Input CSV file containing SMILES strings.
  -o, --output FILE      Output CSV file to save selected SMILES strings.
  -osdf, --output_sdf FILE   Optional SDF file to save selected molecules.
  -mw_min, --mw_min_cutoff FLOAT Minimum molecular weight cutoff for filtering molecules.
  -mw_max, --mw_max_cutoff FLOAT Maximum molecular weight cutoff for filtering molecules.
  -qed, --qed_cutoff FLOAT Optional QED cutoff for filtering molecules.
  -h, --help             Show this help message and exit.

Description:
  This script processes a CSV file of SMILES strings and selects those with
  both known chiral centers and free amines, excluding aromatic NH, acids, and hydroxy groups.
  It can also apply molecular weight and QED cutoffs to filter molecules.
"""
    print(help_message)


def main():
    parser = argparse.ArgumentParser(description="Filter SMILES with free amine, chiral centers, excluding aromatic NH, acids, and hydroxy groups.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Input CSV file containing SMILES strings.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output CSV file to save selected SMILES strings.')
    parser.add_argument('-osdf', '--output_sdf', type=str, default=None, help='Optional SDF file to save selected molecules.')
    parser.add_argument('-mw_min', '--mw_min_cutoff', type=float, default=None, help='Minimum molecular weight cutoff.')
    parser.add_argument('-mw_max', '--mw_max_cutoff', type=float, default=None, help='Maximum molecular weight cutoff.')
    parser.add_argument('-qed', '--qed_cutoff', type=float, default=None, help='Optional QED cutoff for filtering molecules.')
    args = parser.parse_args()

    # Validate input arguments
    if not args.input or not args.output:
        print_help()
        return

    # Process the SMILES
    smiles_list = read_smiles_from_csv(args.input)
    selected_smiles, selected_mols = search_smiles(smiles_list, args.mw_min_cutoff, args.mw_max_cutoff, args.qed_cutoff)

    # Save the selected SMILES to a CSV file
    write_smiles_to_csv(args.output, selected_smiles)

    # Save the selected molecules to an SDF file if requested
    if args.output_sdf:
        write_smiles_to_sdf(args.output_sdf, selected_mols)

    print(f"Selected SMILES have been saved to {args.output}")
    if args.output_sdf:
        print(f"Selected molecules have been saved to {args.output_sdf}")


if __name__ == '__main__':
    main()

