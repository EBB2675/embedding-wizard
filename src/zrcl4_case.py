import numpy as np
from ase import Atoms
from ase.io import read
from scipy.spatial import cKDTree
import nglview as nv


class QCExtractor:
    def __init__(self, cif_file):
        self.cif_file = cif_file
        self.supercell = None
        self.qc_atoms = None
        self.ecp_atoms = None
        self.pc_atoms = None

    def create_supercell(self, scaling_factors=(1, 1, 1)):
        """Create a supercell from the CIF file."""
        self.supercell = read(self.cif_file)
        self.supercell *= scaling_factors
        print(f"Supercell created with {len(self.supercell)} atoms.")

    def extract_qc_region(self, center_atom_symbol="Zr", bonded_symbol="Cl", bonding_threshold=3.0):
        """Extract the QC region (ZrCl6) and center it in the supercell."""
        supercell_positions = self.supercell.get_positions()
        supercell_cell = self.supercell.get_cell()

        # Compute the geometric center of the supercell
        supercell_center = np.sum(supercell_cell, axis=0) / 2.0

        # Find the Zr atom closest to the center
        min_distance = float('inf')
        center_atom_index = None
        for i, atom in enumerate(self.supercell):
            if atom.symbol == center_atom_symbol:
                distance = np.linalg.norm(atom.position - supercell_center)
                if distance < min_distance:
                    min_distance = distance
                    center_atom_index = i

        if center_atom_index is None:
            raise ValueError(f"No {center_atom_symbol} atom found in the supercell.")

        # Find bonded Cl atoms
        tree = cKDTree(supercell_positions, boxsize=supercell_cell.lengths())
        center_atom_position = supercell_positions[center_atom_index]
        bonded_indices = tree.query_ball_point(center_atom_position, r=bonding_threshold)
        bonded_atoms = [self.supercell[i] for i in bonded_indices if self.supercell[i].symbol == bonded_symbol]

        if len(bonded_atoms) != 6:
            raise ValueError(f"Expected 6 bonded {bonded_symbol} atoms, found {len(bonded_atoms)}.")

        # Create the QC region (Zr + 6 Cl atoms)
        qc_atoms = [self.supercell[center_atom_index]] + bonded_atoms
        self.qc_atoms = Atoms(qc_atoms, cell=self.supercell.get_cell(), pbc=True)

        # Debug: Validate QC region placement
        qc_center = self.qc_atoms.get_center_of_mass()
        print(f"QC region extracted with {len(self.qc_atoms)} atoms.")
        print(f"QC region centered at {qc_center}.")

    def assign_layers(self, ecp_radius=3.5):
        """Assign atoms to ECP and PC layers."""
        if self.qc_atoms is None:
            raise ValueError("QC region not initialized. Run extract_qc_region() first.")

        # Get QC positions
        qc_positions = self.qc_atoms.get_positions()
        supercell_positions = self.supercell.get_positions()
        supercell_cell = self.supercell.get_cell()

        # Align positions to the periodic box
        supercell_positions = supercell_positions % supercell_cell.lengths()
        qc_positions = qc_positions % supercell_cell.lengths()

        # Use KDTree to classify atoms
        tree = cKDTree(supercell_positions, boxsize=supercell_cell.lengths())

        # Identify QC indices
        qc_indices = set()
        for qc_pos in qc_positions:
            closest_index = tree.query(qc_pos)[1]  # Get the closest atom index
            qc_indices.add(closest_index)

        # Identify ECP indices
        ecp_indices = set()
        for qc_pos in qc_positions:
            neighbors = tree.query_ball_point(qc_pos, r=ecp_radius)
            ecp_indices.update(neighbors)

        ecp_indices -= qc_indices  # Ensure no overlap between QC and ECP

        # Assign PC layer: All remaining atoms
        all_indices = set(range(len(self.supercell)))
        pc_indices = all_indices - qc_indices - ecp_indices

        # Debugging: Print counts and overlaps
        print(f"QC indices: {len(qc_indices)}, ECP indices: {len(ecp_indices)}, PC indices: {len(pc_indices)}")
        print(f"QC & ECP overlap: {len(qc_indices & ecp_indices)}, ECP & PC overlap: {len(ecp_indices & pc_indices)}")

        # Ensure all atoms are assigned exactly once
        total_classified_atoms = len(qc_indices) + len(ecp_indices) + len(pc_indices)
        assert total_classified_atoms == len(self.supercell), (
            f"Discrepancy in atom count! Total classified: {total_classified_atoms}, "
            f"Expected: {len(self.supercell)}"
        )

        # Create ECP and PC layers
        ecp_atoms_list = [self.supercell[i] for i in ecp_indices]
        pc_atoms_list = [self.supercell[i] for i in pc_indices]

        # Assign layers
        self.ecp_atoms = Atoms(ecp_atoms_list, cell=self.supercell.get_cell(), pbc=True)
        self.pc_atoms = Atoms(pc_atoms_list, cell=self.supercell.get_cell(), pbc=True)

        print(f"QC region: {len(self.qc_atoms)} atoms, ECP layer: {len(self.ecp_atoms)} atoms, PC layer: {len(self.pc_atoms)} atoms.")

    def get_regions(self):
        """Return the QC, ECP, and PC regions."""
        return self.qc_atoms, self.ecp_atoms, self.pc_atoms

    def visualize_regions(self):
        """Visualize the QC, ECP, and PC regions using nglview."""
        view_qc = nv.show_ase(self.qc_atoms)
        view_ecp = nv.show_ase(self.ecp_atoms)
        view_pc = nv.show_ase(self.pc_atoms)
        return view_qc, view_ecp, view_pc

