from ase.visualize import view
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
import numpy as np
import nglview as nv
from IPython.display import display, HTML


class SupercellCreator:
    def __init__(self, cif_file, x_scaling, y_scaling, z_scaling):
        self.structure = read(cif_file)  # Load the structure from the CIF file
        self.scaling_matrix = (x_scaling, y_scaling, z_scaling)  # Create the scaling matrix as a tuple
        self.supercell = self.make_supercell()  # Create the supercell

    def make_supercell(self):
        """Create a supercell based on the given scaling factors."""
        return self.structure.repeat(self.scaling_matrix)  # Use the scaling factors directly

    def get_cell_center(self):
        """Calculate the center of the supercell."""
        return self.supercell.get_center_of_mass()  # Get the center based on the mass of the atoms

    def calculate_radii(self):
        """Calculate the radii for different layers based on the supercell dimensions."""
        cell = self.supercell.get_cell()  # Get the cell vectors
        cell_length = np.max(np.linalg.norm(cell, axis=1))  # Calculate lengths of the cell vectors

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
        """Retrieve atoms in each defined layer."""
        qc_atoms_list = []
        ecp_atoms_list = []
        pc_atoms_list = []

        cell_center = self.get_cell_center()  # Get the center of the supercell
        qc_radius, ecp_radius, pc_radius = self.calculate_radii()  # Calculate the radii

        # Select the QC atoms
        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            # If the atom is within the QC radius, add it to qc_atoms_list
            if distance <= qc_radius:
                qc_atoms_list.append(atom)

        # Now, determine the ECP atoms (outside QC but within ECP radius)
        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            # Ensure ECP atoms are outside the QC radius but within ECP radius
            if qc_radius < distance <= ecp_radius:
                ecp_atoms_list.append(atom)

        # Now, determine the PC atoms (outside ECP radius)
        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            # Ensure PC atoms are outside the ECP radius
            if distance > ecp_radius:
                pc_atoms_list.append(atom)

        # Convert lists of atoms back to Atoms objects
        qc_atoms = Atoms(qc_atoms_list, cell=self.supercell.get_cell(), pbc=True)
        ecp_atoms = Atoms(ecp_atoms_list, cell=self.supercell.get_cell(), pbc=True)
        pc_atoms = Atoms(pc_atoms_list, cell=self.supercell.get_cell(), pbc=True)

        return qc_atoms, ecp_atoms, pc_atoms  # Return the Atoms objects for each layer


