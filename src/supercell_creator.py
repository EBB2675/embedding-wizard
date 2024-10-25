from ase.visualize import view
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
import numpy as np
from collections import Counter
import nglview as nv


class SupercellCreator:
    def __init__(self, cif_file, x_scaling=1, y_scaling=1, z_scaling=1, qc_radius=None, ecp_radius=None, pc_radius=None):
        # Read CIF file and initialize structure
        self.structure = read(cif_file)
        self.scaling_matrix = (x_scaling, y_scaling, z_scaling)
        self.supercell = self.make_supercell()
        self.stoichiometry = self.get_stoichiometry()

        # Set radii based on defaults if not provided
        self.qc_radius, self.ecp_radius, self.pc_radius = self.initialize_radii(qc_radius, ecp_radius, pc_radius)

    def make_supercell(self):
        """Create the supercell using the scaling factors provided."""
        return self.structure.repeat(self.scaling_matrix)

    def get_stoichiometry(self):
        """Determine and return the stoichiometry of the original structure."""
        return Counter(atom.symbol for atom in self.structure)

    def initialize_radii(self, qc_radius, ecp_radius, pc_radius):
        """Initialize default radii for QC, ECP, and PC layers if not provided."""
        cell_length = np.max(np.linalg.norm(self.supercell.get_cell(), axis=1))
        # Calculate default radii based on cell length if not provided
        qc_radius = qc_radius or cell_length * 0.10
        ecp_radius = ecp_radius or qc_radius + cell_length * 0.10
        pc_radius = pc_radius or ecp_radius + cell_length * 0.70
        return qc_radius, ecp_radius, pc_radius

    def display_molecule_in_unit_cell(self):
        """Display the molecule inside the unit cell using NGLView."""
        print("Displaying the original unit cell with the molecule...")
        view_widget = nv.show_ase(self.structure)  # Use NGLView to show the structure
        return view_widget  # Return the widget for display in Jupyter

    def expand_supercell_based_on_radius(self, radius):
        """Create a supercell view with atoms within a given radius from the center."""
        cell_center = self.supercell.get_center_of_mass()
        atoms_within_radius = [atom for atom in self.supercell if np.linalg.norm(atom.position - cell_center) <= radius]
        expanded_supercell = Atoms(atoms_within_radius, cell=self.supercell.get_cell(), pbc=True)

        print(f"Displaying supercell expanded to radius {radius}.")
        view_widget = nv.show_ase(expanded_supercell)  # Use NGLView here as well
        return view_widget  # Return the widget

    def get_atoms_in_layers(self):
        """Classify atoms into QC, ECP, and PC layers based on radii."""
        cell_center = self.supercell.get_center_of_mass()
        qc_atoms, ecp_atoms, pc_atoms = [], [], []

        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            if distance <= self.qc_radius:
                qc_atoms.append(atom)
            elif self.qc_radius < distance <= self.ecp_radius:
                ecp_atoms.append(atom)
            else:
                pc_atoms.append(atom)

        print(f"QC layer: {len(qc_atoms)} atoms, {dict(Counter(atom.symbol for atom in qc_atoms))}")
        print(f"ECP layer: {len(ecp_atoms)} atoms, {dict(Counter(atom.symbol for atom in ecp_atoms))}")
        print(f"PC layer: {len(pc_atoms)} atoms, {dict(Counter(atom.symbol for atom in pc_atoms))}")

        return (Atoms(qc_atoms, cell=self.supercell.get_cell(), pbc=True),
                Atoms(ecp_atoms, cell=self.supercell.get_cell(), pbc=True),
                Atoms(pc_atoms, cell=self.supercell.get_cell(), pbc=True))

# Usage example:
# Ensure you are in a Jupyter Notebook to visualize the output
# creator = SupercellCreator('path/to/your/file.cif')
# view_widget = creator.display_molecule_in_unit_cell()
# view_widget  # Execute this line to show the NGLView widget

