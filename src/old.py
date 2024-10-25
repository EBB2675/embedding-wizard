from ase.visualize import view
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
import numpy as np
import nglview as nv
from IPython.display import display, HTML
from collections import Counter
import ipywidgets as widgets


class SupercellCreator:
    def __init__(self, cif_file, x_scaling, y_scaling, z_scaling, qc_radius=None, ecp_radius=None, pc_radius=None):
        self.structure = read(cif_file)
        self.scaling_matrix = (x_scaling, y_scaling, z_scaling)
        self.supercell = self.make_supercell()
        self.stoichiometry = self.get_stoichiometry()

        # Use provided radii or calculate default values based on cell size
        self.qc_radius = qc_radius or self.calculate_default_radii()[0]
        self.ecp_radius = ecp_radius or self.calculate_default_radii()[1]
        self.pc_radius = pc_radius or self.calculate_default_radii()[2]

        # Store selected QC atoms
        self.selected_qc_atoms = []

    def make_supercell(self):
        """Create a supercell based on the scaling factors."""
        return self.structure.repeat(self.scaling_matrix)

    def get_stoichiometry(self):
        """Determine the stoichiometry of the structure."""
        element_counts = Counter(atom.symbol for atom in self.structure)
        return element_counts

    def calculate_default_radii(self, qc_scale=0.10, ecp_scale=0.10, pc_scale=0.7):
        """Calculate default radii for different layers based on the supercell dimensions and scaling factors."""
        cell = self.supercell.get_cell()
        cell_length = np.max(np.linalg.norm(cell, axis=1))
        qc_radius = cell_length * qc_scale
        ecp_radius = qc_radius + (cell_length * ecp_scale)
        pc_radius = ecp_radius + (cell_length * pc_scale)
        return qc_radius, ecp_radius, pc_radius

    def display_unit_cell(self):
        """Display the basic unit cell."""
        print("Displaying the original unit cell...")
        view(self.structure)  # ASE viewer for unit cell visualization

    def expand_supercell_based_on_distance(self, radius):
        """Expand the unit cell based on a user-defined distance."""
        cell_center = self.supercell.get_center_of_mass()
        expanded_atoms = [atom for atom in self.supercell if np.linalg.norm(atom.position - cell_center) <= radius]
        expanded_supercell = Atoms(expanded_atoms, cell=self.supercell.get_cell(), pbc=True)
        
        print(f"Supercell expanded to radius {radius}.")
        view(expanded_supercell)
    
        return expanded_supercell

    def decide_qc_layer(self, radius):
        """Define the QC layer based on a radius."""
        self.qc_radius = radius
        cell_center = self.supercell.get_center_of_mass()
        qc_atoms = [atom for atom in self.supercell if np.linalg.norm(atom.position - cell_center) <= self.qc_radius]
        qc_supercell = Atoms(qc_atoms, cell=self.supercell.get_cell(), pbc=True)
        
        print(f"QC layer defined with radius {radius}.")
        view(qc_supercell)
        
        return qc_supercell

    def decide_ecp_layer(self, ecp_radius):
        """Define the ECP layer based on a radius from the QC layer."""
        if not self.qc_radius:
            raise ValueError("QC layer must be defined before the ECP layer.")
        
        self.ecp_radius = ecp_radius
        cell_center = self.supercell.get_center_of_mass()
        ecp_atoms = [atom for atom in self.supercell if self.qc_radius < np.linalg.norm(atom.position - cell_center) <= self.ecp_radius]
        ecp_supercell = Atoms(ecp_atoms, cell=self.supercell.get_cell(), pbc=True)
        
        print(f"ECP layer defined with radius {ecp_radius} from QC.")
        view(ecp_supercell)

        return ecp_supercell

    def assign_remaining_to_pc_layer(self):
        """Assign all atoms outside the ECP radius to the PC layer."""
        if not self.ecp_radius:
            raise ValueError("ECP layer must be defined before the PC layer.")

        cell_center = self.supercell.get_center_of_mass()
        pc_atoms = [atom for atom in self.supercell if np.linalg.norm(atom.position - cell_center) > self.ecp_radius]
        pc_supercell = Atoms(pc_atoms, cell=self.supercell.get_cell(), pbc=True)
        
        print(f"PC layer defined with all atoms outside ECP.")
        view(pc_supercell)

        return pc_supercell

    def show_interactive_supercell(self):
        """Display the supercell interactively with NGLView, allowing atom selection for the QC layer."""
        view = nv.show_ase(self.supercell)
        view.add_representation('ball+stick')
        view.add_representation('label', selection='*')  # Show labels for easier selection
        
        def on_click_pick(selection):
            """Callback for atom selection."""
            atom_index = selection[0]['atom_index']
            atom = self.supercell[atom_index]
            print(f"Selected atom: {atom.symbol}, index: {atom_index}, position: {atom.position}")

            if atom_index not in self.selected_qc_atoms:
                self.selected_qc_atoms.append(atom_index)
                print(f"Atom {atom.symbol} added to QC.")
            else:
                self.selected_qc_atoms.remove(atom_index)
                print(f"Atom {atom.symbol} removed from QC.")

        view.observe(on_click_pick, names='picked')
        display(view)

    def get_selected_qc_atoms(self):
        """Return the selected QC atoms."""
        qc_atoms = Atoms([self.supercell[i] for i in self.selected_qc_atoms], cell=self.supercell.get_cell(), pbc=True)
        return qc_atoms

