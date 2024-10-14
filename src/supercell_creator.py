from ase.visualize import view
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
import numpy as np
import nglview as nv
from IPython.display import display, HTML
from collections import Counter

class SupercellCreator:
    def __init__(self, cif_file, x_scaling, y_scaling, z_scaling):
        self.structure = read(cif_file)  # Load the structure from the CIF file
        self.scaling_matrix = (x_scaling, y_scaling, z_scaling)  # Scaling factors
        self.supercell = self.make_supercell()  # Create the supercell
        self.stoichiometry = self.get_stoichiometry()  # Get the stoichiometry from the CIF file

    def make_supercell(self):
        """Create a supercell based on the given scaling factors."""
        return self.structure.repeat(self.scaling_matrix)  # Repeat the unit cell

    def get_cell_center(self):
        """Calculate the center of the supercell."""
        return self.supercell.get_center_of_mass()  # Get the center based on the mass of the atoms

    def get_stoichiometry(self):
        """Determine the stoichiometry of the structure."""
        element_counts = Counter(atom.symbol for atom in self.structure)  # Count atoms by element
        min_count = min(element_counts.values())  # Find the smallest number of atoms for any element
        stoichiometry = {element: count // min_count for element, count in element_counts.items()}  # Normalize ratios
        return stoichiometry

    def calculate_radii(self):
        """Calculate the radii for different layers based on the supercell dimensions."""
        cell = self.supercell.get_cell()  # Get the cell vectors
        cell_length = np.max(np.linalg.norm(cell, axis=1))  # Calculate the lengths of the cell vectors

        # Define the number of layers and unit cells
        qc_unit_cell_num = 1
        ecp_layer_num = 1

        # Calculate the radii
        qc_radius = cell_length / 10.0 * qc_unit_cell_num  # QC layer radius based on unit cell size
        ecp_radius = qc_radius + (cell_length / 10.0) * ecp_layer_num  # ECP layer radius

        # Calculate the maximum radius for PC layers
        pc_radius = ecp_radius + (cell_length / 10.0 * 5)  # 5 layers of PC, adjust as needed

        return qc_radius, ecp_radius, pc_radius  # Return the calculated radii

    def get_atoms_in_layers(self):
        """Retrieve atoms in each defined layer while retaining stoichiometry in the QC region."""
        qc_atoms_list = []
        ecp_atoms_list = []
        pc_atoms_list = []

        cell_center = self.get_cell_center()  # Get the center of the supercell
        qc_radius, ecp_radius, pc_radius = self.calculate_radii()  # Calculate the radii

        # Create counters to track how many atoms of each type have been added
        element_count = Counter()
        target_ratios = self.stoichiometry  # Stoichiometry ratios obtained from CIF

        # First pass: select the QC atoms based on distance and stoichiometry
        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            if distance <= qc_radius:
                element = atom.symbol
                # Check if adding this atom would maintain the stoichiometric ratio
                total_atoms = sum(element_count.values()) + 1  # Include this atom in the count
                element_target_ratio = target_ratios[element] / sum(target_ratios.values())
                current_ratio = element_count[element] / total_atoms if total_atoms > 0 else 0

                # Add the atom if its current ratio is within the desired stoichiometric ratio
                if current_ratio <= element_target_ratio:
                    qc_atoms_list.append(atom)
                    element_count[element] += 1

        # Second pass: determine the ECP atoms (outside QC but within ECP radius)
        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            if qc_radius < distance <= ecp_radius:
                ecp_atoms_list.append(atom)

        # Third pass: determine the PC atoms (outside ECP radius)
        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            if distance > ecp_radius:
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

