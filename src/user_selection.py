import numpy as np
from ase import Atoms, Atom
from ase.neighborlist import NeighborList
from ase.io import read  # ASE reader for XYZ and CIF files

class UserDefinedQCRegion:
    
    def xyz_to_atoms(self, xyz_file_path):
        """
        Reads an XYZ file and converts it to an ASE Atoms object.

        Parameters:
        - xyz_file_path: Path to the XYZ file.

        Returns:
        - ASE Atoms object representing the molecule.
        """
        try:
            atoms = read(xyz_file_path)
            print(f"Loaded XYZ file with {len(atoms)} atoms.")
            return atoms
        except Exception as e:
            print(f"Error loading XYZ file: {e}")
            return None

    def match_qc_region_to_cif(self, xyz_atoms, cif_file_path, ecp_distance=3.0):
        """
        Matches QC region atoms from XYZ input to CIF structure, and adds an ECP layer around it.

        Parameters:
        - xyz_atoms: ASE Atoms object representing the QC region (from XYZ file).
        - cif_file_path: Path to the CIF file.
        - ecp_distance: Distance threshold to consider an atom as part of the ECP layer.

        Returns:
        - ASE Atoms object with QC region, ECP, and PC layers.
        """
        # Load CIF file
        full_structure = read(cif_file_path)
        print(f"Loaded CIF structure with {len(full_structure)} atoms.")

        # Calculate the center of mass for the QC region
        qc_com = xyz_atoms.get_center_of_mass()

        # Define neighbor list for CIF atoms to find ECP neighbors around QC atoms
        cutoffs = [ecp_distance / 2.0] * len(full_structure)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(full_structure)

        # Initialize lists to track QC, ECP, and PC atoms
        qc_atoms = []
        ecp_atoms = []
        pc_atoms = []

        # Match CIF atoms to QC region atoms based on proximity to the QC region's center of mass
        for atom in full_structure:
            if np.linalg.norm(atom.position - qc_com) <= ecp_distance:
                if any(np.linalg.norm(atom.position - xyz_atom.position) < 1.0 for xyz_atom in xyz_atoms):
                    qc_atoms.append(atom)
                else:
                    ecp_atoms.append(Atom(symbol=atom.symbol, position=atom.position))
            else:
                pc_atoms.append(Atom(symbol='PC', position=atom.position))

        # Combine QC, ECP, and PC atoms into a single Atoms object
        combined_atoms = Atoms(qc_atoms + ecp_atoms + pc_atoms, cell=full_structure.get_cell(), pbc=True)
        print(f"QC Region: {len(qc_atoms)}, ECP Layer: {len(ecp_atoms)}, PC Layer: {len(pc_atoms)}")
        
        return combined_atoms

