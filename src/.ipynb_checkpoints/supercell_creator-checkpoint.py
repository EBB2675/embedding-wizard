import nglview as nv
from ase.io import write
from ase import Atoms

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

        # Store the selected QC atoms
        self.selected_qc_atoms = []

    def make_supercell(self):
        """Create a supercell based on the given scaling factors."""
        return self.structure.repeat(self.scaling_matrix)  # Repeat the unit cell

    def get_stoichiometry(self):
        """Determine the stoichiometry of the structure."""
        element_counts = Counter(atom.symbol for atom in self.structure)  # Count atoms by element
        return element_counts

    def calculate_default_radii(self, qc_scale=0.125, ecp_scale=0.125, pc_scale=0.625):
        """Calculate default radii for different layers based on the supercell dimensions and scaling factors."""
        cell = self.supercell.get_cell()  # Get the cell vectors
        cell_length = np.max(np.linalg.norm(cell, axis=1))  # Calculate the lengths of the cell vectors

        # Calculate the default radii using scaling factors
        qc_radius = cell_length * qc_scale  # QC layer radius based on scale factor
        ecp_radius = qc_radius + (cell_length * ecp_scale)  # ECP layer radius based on scale factor
        pc_radius = ecp_radius + (cell_length * pc_scale)  # PC layer radius based on scale factor

        return qc_radius, ecp_radius, pc_radius

    def show_interactive_supercell(self):
        """
        Display the initial supercell interactively in NGLView and allow the user to select QC atoms.
        """
        view = nv.show_ase(self.supercell)  # Visualize the supercell using NGLView
        view.add_representation('ball+stick')  # Add ball-and-stick representation
        view.add_representation('label', selection='*')  # Show labels for atoms to help with selection

        # Enable interactive atom selection
        def on_click_pick(selection):
            """Callback function when an atom is clicked."""
            atom_index = selection[0]['atom_index']
            atom = self.supercell[atom_index]
            print(f"Selected atom: {atom.symbol}, index: {atom_index}, position: {atom.position}")

            if atom_index not in self.selected_qc_atoms:
                self.selected_qc_atoms.append(atom_index)
                print(f"Atom {atom.symbol} added to QC region.")
            else:
                self.selected_qc_atoms.remove(atom_index)
                print(f"Atom {atom.symbol} removed from QC region.")

        view.observe(on_click_pick, names='picked')

        display(view)  # Show the NGLView widget in Jupyter

    def get_selected_qc_atoms(self):
        """Return the atoms selected for the QC region."""
        qc_atoms = Atoms([self.supercell[i] for i in self.selected_qc_atoms], cell=self.supercell.get_cell(), pbc=True)
        return qc_atoms

    def get_atoms_in_layers(self):
        """Retrieve atoms in each defined layer, with interactive selection for QC atoms."""
        qc_atoms = self.get_selected_qc_atoms()  # Get user-selected QC atoms
        cell_center = self.supercell.get_center_of_mass()  # Get the center of the supercell
        
        ecp_atoms_list = []
        pc_atoms_list = []

        # Second pass: determine the ECP atoms (outside QC but within ECP radius)
        for atom in self.supercell:
            if atom.index not in self.selected_qc_atoms:
                distance = np.linalg.norm(atom.position - cell_center)
                if self.qc_radius < distance <= self.ecp_radius:
                    ecp_atoms_list.append(atom)

        # Third pass: determine the PC atoms (outside ECP radius)
        for atom in self.supercell:
            if atom.index not in self.selected_qc_atoms:
                distance = np.linalg.norm(atom.position - cell_center)
                if distance > self.ecp_radius:
                    pc_atoms_list.append(atom)

        # Convert lists of atoms back to Atoms objects
        ecp_atoms = Atoms(ecp_atoms_list, cell=self.supercell.get_cell(), pbc=True)
        pc_atoms = Atoms(pc_atoms_list, cell=self.supercell.get_cell(), pbc=True)

        # Get individual atom counts
        qc_counts = Counter(qc_atoms.get_chemical_symbols())
        ecp_counts = Counter(ecp_atoms.get_chemical_symbols())
        pc_counts = Counter(pc_atoms.get_chemical_symbols())

        # Print total atom counts and individual atom counts
        print(f"QC region: {len(qc_atoms)} atoms")
        print(f"Individual counts: {dict(qc_counts)}")

        print(f"ECP region: {len(ecp_atoms)} atoms")
        print(f"Individual counts: {dict(ecp_counts)}")

        print(f"PC region: {len(pc_atoms)} atoms")
        print(f"Individual counts: {dict(pc_counts)}")

        return qc_atoms, ecp_atoms, pc_atoms  # Return the Atoms objects for each layer
