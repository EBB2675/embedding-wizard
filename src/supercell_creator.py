from ase.visualize import view
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
import numpy as np
import nglview as nv
from IPython.display import display, HTML
from collections import Counter


class SupercellCreator:
    def __init__(self, cif_file, x_scaling, y_scaling, z_scaling, qc_radius=None, ecp_radius=None, pc_radius=None):
        self.structure = read(cif_file) 
        self.scaling_matrix = (x_scaling, y_scaling, z_scaling)  
        self.supercell = self.make_supercell()  
        self.stoichiometry = self.get_stoichiometry()  

        # Use provided radii or default values based on cell size
        self.qc_radius = qc_radius or self.calculate_default_radii()[0]
        self.ecp_radius = ecp_radius or self.calculate_default_radii()[1]
        self.pc_radius = pc_radius or self.calculate_default_radii()[2]

    def make_supercell(self):
        """Create a supercell based on the given scaling factors."""
        return self.structure.repeat(self.scaling_matrix)  # Repeat the unit cell

    def get_cell_center(self):
        """Calculate the center of the supercell."""
        return self.supercell.get_center_of_mass()  # Get the center based on the mass of the atoms

    def get_stoichiometry(self):
        """Determine the stoichiometry of the structure."""
        element_counts = Counter(atom.symbol for atom in self.structure)  # Count atoms by element
        return element_counts  

    def calculate_default_radii(self):
        """Calculate default radii for different layers based on the supercell dimensions."""
        cell = self.supercell.get_cell()  # Get the cell vectors
        cell_length = np.max(np.linalg.norm(cell, axis=1))  # Calculate the lengths of the cell vectors

        # Calculate the default radii
        qc_radius = cell_length / 8.0  # Default QC layer radius based on unit cell size
        ecp_radius = qc_radius + (cell_length / 8.0)  # Default ECP layer radius
        pc_radius = ecp_radius + (cell_length / 8.0 * 5)  # Default PC layer radius

        return qc_radius, ecp_radius, pc_radius  

    def get_atoms_in_layers(self):
        """Retrieve atoms in each defined layer while retaining stoichiometry in the QC region."""
        qc_atoms_list = []
        ecp_atoms_list = []
        pc_atoms_list = []

        cell_center = self.get_cell_center()  # Get the center of the supercell

        # Create counters to track how many atoms of each type have been added
        element_count = Counter()
        target_ratios = self.stoichiometry  # Stoichiometry ratios obtained from CIF

        # Calculate total number of QC atoms required based on stoichiometry
        total_qc_atoms = sum(target_ratios.values())
        qc_atom_counts = {element: int((count / total_qc_atoms) * sum(target_ratios.values())) for element, count in target_ratios.items()}

        # First pass: collect distances and sort atoms by distance
        distances = []
        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            distances.append((distance, atom))

        # Sort atoms by distance to the cell center
        distances.sort(key=lambda x: x[0])

        # Select atoms while ensuring the correct stoichiometry
        for distance, atom in distances:
            if distance <= self.qc_radius and element_count[atom.symbol] < qc_atom_counts.get(atom.symbol, 0):
                qc_atoms_list.append(atom)
                element_count[atom.symbol] += 1

        # Second pass: determine the ECP atoms (outside QC but within ECP radius)
        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            if self.qc_radius < distance <= self.ecp_radius:
                ecp_atoms_list.append(atom)

        # Third pass: determine the PC atoms (outside ECP radius)
        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            if distance > self.ecp_radius:
                pc_atoms_list.append(atom)

        # Convert lists of atoms back to Atoms objects
        qc_atoms = Atoms(qc_atoms_list, cell=self.supercell.get_cell(), pbc=True)
        ecp_atoms = Atoms(ecp_atoms_list, cell=self.supercell.get_cell(), pbc=True)
        pc_atoms = Atoms(pc_atoms_list, cell=self.supercell.get_cell(), pbc=True)

        # Get individual atom counts
        qc_counts = Counter(qc_atoms.get_chemical_symbols())
        ecp_counts = Counter(ecp_atoms.get_chemical_symbols())
        pc_counts = Counter(pc_atoms.get_chemical_symbols())

        # Print total atom counts and individual atom counts
        print(f"QC region: {len(qc_atoms_list)} atoms")
        print(f"Individual counts: {dict(qc_counts)}")

        print(f"ECP region: {len(ecp_atoms_list)} atoms")
        print(f"Individual counts: {dict(ecp_counts)}")

        print(f"PC region: {len(pc_atoms_list)} atoms")
        print(f"Individual counts: {dict(pc_counts)}")

        return qc_atoms, ecp_atoms, pc_atoms  # Return the Atoms objects for each layer

