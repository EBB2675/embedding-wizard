import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import NeighborList  # Ensure the correct import
import nglview as nv

class UserDefinedQCRegion:
    def __init__(self, cif_file, xyz_file):
        self.cif_file = cif_file
        self.xyz_file = xyz_file
        self.qc_atoms = None
        self.supercell = None
        self.ecp_atoms = None
        self.pc_atoms = None
    
    def read_xyz_to_atoms(self):
        """Read the XYZ file and create an Atoms object for the QC region."""
        self.qc_atoms = read(self.xyz_file)
        print(f"QC region loaded with {len(self.qc_atoms)} atoms.")

    def create_supercell(self, scaling_factors=(1, 1, 1)):
        """Create a supercell from the CIF file."""
        self.supercell = read(self.cif_file)
        self.supercell *= scaling_factors
        print(f"Supercell created with {len(self.supercell)} atoms.")

    def classify_atoms(self, ecp_distance=2.5):
        """Classify atoms into QC, ECP, and PC layers."""
        if self.qc_atoms is None or self.supercell is None:
            raise ValueError("QC atoms or supercell not initialized.")

        # Get positions of QC atoms
        qc_positions = self.qc_atoms.get_positions()
        ecp_atoms_list = []
        pc_atoms_list = []

        # Get positions of supercell atoms
        positions = self.supercell.get_positions()

        # Create a NeighborList for the supercell atoms
        cutoffs = [ecp_distance] * len(positions)  # One cutoff for each atom in the supercell
        neighbor_list = NeighborList(cutoffs, skin=0.3, sorted=False, self_interaction=True)
        # the following is dangerous:
        neighbor_list.update(self.supercell)  # Update with supercell atom positions

        for i, atom in enumerate(self.supercell):
            # Check if the current atom is in the QC layer
            if atom in self.qc_atoms:
                pc_atoms_list.append(atom)
                continue  # Skip QC atoms for further checking

            # Get neighbors for the current atom with respect to all atoms in the supercell
            indices, offsets = neighbor_list.get_neighbors(i)

            # Only consider neighbors that are valid indices in qc_positions
            valid_indices = [idx for idx in indices if idx < len(qc_positions)]

            # Check if any of the valid neighbors are in the QC region
            if valid_indices and np.any(np.linalg.norm(qc_positions[valid_indices] - atom.position, axis=1) < ecp_distance):
                ecp_atoms_list.append(atom)
            else:
                pc_atoms_list.append(atom)

        self.ecp_atoms = Atoms(ecp_atoms_list)
        self.pc_atoms = Atoms(pc_atoms_list)

        print(f"QC layer: {len(self.qc_atoms)} atoms, ECP layer: {len(self.ecp_atoms)} atoms, PC layer: {len(self.pc_atoms)} atoms.")

    def get_regions(self):
        """Return the QC, ECP, and PC regions."""
        return self.qc_atoms, self.ecp_atoms, self.pc_atoms

