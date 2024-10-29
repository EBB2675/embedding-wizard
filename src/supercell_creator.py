from ase.visualize import view
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
import numpy as np
from collections import Counter
import nglview as nv
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform

class SupercellCreator:
    def __init__(self, cif_file, x_scaling=1, y_scaling=1, z_scaling=1, qc_radius=None, ecp_radius=None, pc_radius=None, bond_distance_threshold=2.7):
        self.structure = read(cif_file)
        self.scaling_matrix = (x_scaling, y_scaling, z_scaling)
        self.supercell = self.make_supercell()
        self.stoichiometry = self.get_stoichiometry()
        self.qc_radius, self.ecp_radius, self.pc_radius = self.initialize_radii(qc_radius, ecp_radius, pc_radius)
        self.bond_distance_threshold = bond_distance_threshold

    def make_supercell(self):

        return self.structure.repeat(self.scaling_matrix)

    def get_stoichiometry(self):
        return Counter(atom.symbol for atom in self.structure)

    def initialize_radii(self, qc_radius, ecp_radius, pc_radius):
        cell_length = np.max(np.linalg.norm(self.supercell.get_cell(), axis=1))
        qc_radius = qc_radius or cell_length * 0.10
        ecp_radius = ecp_radius or qc_radius + cell_length * 0.10
        pc_radius = pc_radius or ecp_radius + cell_length * 0.70
        return qc_radius, ecp_radius, pc_radius

    def get_atoms_in_layers(self, charge_dict):
        cell_center = self.supercell.get_center_of_mass()
        qc_atoms, ecp_atoms, pc_atoms = [], [], []
        qc_charge = 0  # Initialize charge for the QC layer

        # Collect QC atoms based on distance
        for atom in self.supercell:
            distance = np.linalg.norm(atom.position - cell_center)
            atom_charge = charge_dict.get(atom.symbol, 0)

            if distance <= self.qc_radius:
                qc_atoms.append(atom)
                qc_charge += atom_charge
            elif self.qc_radius < distance <= self.ecp_radius:
                ecp_atoms.append(atom)
            else:
                pc_atoms.append(atom)

        # Ensure QC layer atoms are directly bonded
        qc_atoms = self.ensure_bonded(qc_atoms)

        # Finalize the QC layer to ensure it is neutral
        qc_atoms, qc_charge = self.finalize_qc_layer(qc_atoms, charge_dict)

        print(f"QC layer: {len(qc_atoms)} atoms, Charge: {qc_charge}, Composition: {dict(Counter(atom.symbol for atom in qc_atoms))}")
        print(f"ECP layer: {len(ecp_atoms)} atoms, Composition: {dict(Counter(atom.symbol for atom in ecp_atoms))}")
        print(f"PC layer: {len(pc_atoms)} atoms, Composition: {dict(Counter(atom.symbol for atom in pc_atoms))}")

        return (Atoms(qc_atoms, cell=self.supercell.get_cell(), pbc=True),
                Atoms(ecp_atoms, cell=self.supercell.get_cell(), pbc=True),
                Atoms(pc_atoms, cell=self.supercell.get_cell(), pbc=True))

    def display_unit_cell(self):
        """Display the basic unit cell."""
        print("Displaying the original unit cell...")
        view = nv.show_ase(self.structure)
        return view

    def ensure_bonded(self, atoms):
        if len(atoms) < 2:
            return atoms  # No need to check if 0 or 1 atom

        atom_positions = np.array([atom.position for atom in atoms])
        dist_matrix = distance_matrix(atom_positions, atom_positions)

        # Identify bonded pairs based on the bond distance threshold
        bonded_indices = np.any(dist_matrix < self.bond_distance_threshold, axis=1)
        bonded_atoms = [atom for i, atom in enumerate(atoms) if bonded_indices[i]]

        return bonded_atoms  # Return the bonded atoms

    def finalize_qc_layer(self, qc_atoms, charge_dict):
        total_charge = sum(charge_dict.get(atom.symbol, 0) for atom in qc_atoms)

        print(f"Initial QC Charge: {total_charge}")

        # While the QC layer is not charge neutral
        while total_charge != 0:
            if total_charge > 0:  # Too positive, need to add negative charge
                added_atom = self.get_bonded_atoms(qc_atoms, charge_dict, negative=True)
                if added_atom:
                    qc_atoms.append(added_atom)
                    total_charge += charge_dict.get(added_atom.symbol, 0)
                    print(f"Added negative atom: {added_atom.symbol}, New Charge: {total_charge}")
                else:
                    break  # No more negatively charged atoms to add
            elif total_charge < 0:  # Too negative, need to remove negatively charged atoms
                self.remove_negatively_charged_atoms(qc_atoms, charge_dict)
                total_charge = sum(charge_dict.get(atom.symbol, 0) for atom in qc_atoms)  # Recalculate total charge
                print(f"Removed negatively charged atoms. New QC Charge: {total_charge}")

        # Ensure the added atoms are also bonded
        qc_atoms = self.ensure_bonded(qc_atoms)

        print(f"Final QC Charge after adjustments: {total_charge}")

        return qc_atoms, total_charge

    def get_bonded_atoms(self, qc_atoms, charge_dict, negative=True):
        """Get bonded atoms from the supercell that can balance charge."""
        current_qc_positions = np.array([atom.position for atom in qc_atoms])
        potential_atoms = []

        for atom in self.supercell:
            atom_charge = charge_dict.get(atom.symbol, 0)
            distance_to_qc = np.linalg.norm(atom.position - current_qc_positions)

            # Check if atom is directly bonded and contributes to charge balancing
            if (negative and atom_charge < 0 and distance_to_qc < self.bond_distance_threshold) or \
               (not negative and atom_charge > 0 and distance_to_qc < self.bond_distance_threshold):
                potential_atoms.append(atom)

        if potential_atoms:
            return potential_atoms[0]  # Return the first potential atom

        return None  # No atom available

    def remove_negatively_charged_atoms(self, qc_atoms, charge_dict):
        """Remove the minimum number of negatively charged atoms to achieve charge neutrality."""
        negatively_charged_atoms = [atom for atom in qc_atoms if charge_dict.get(atom.symbol, 0) < 0]

        # Sort negatively charged atoms by charge magnitude (more negative first)
        negatively_charged_atoms.sort(key=lambda atom: charge_dict.get(atom.symbol, 0))

        # Calculate the charge needed to reach neutrality
        charge_needed = sum(charge_dict.get(atom.symbol, 0) for atom in qc_atoms)

        # Remove the minimum number of negatively charged atoms until the charge is neutralized
        for atom in negatively_charged_atoms:
            if charge_needed < 0:  # Still need to remove charge
                qc_atoms.remove(atom)
                charge_needed -= charge_dict.get(atom.symbol, 0)  # Update the charge needed
                print(f"Removed negatively charged atom: {atom.symbol}")

        return qc_atoms  # Return the modified list of QC atoms

    def isolate_unit_molecule(self, center_atom_symbol='Zr', bonded_symbol='Cl', bonding_threshold=2.7,
                              exact_bonded_atoms=4, max_retries=25):
        """
        Isolate a single unit molecule (e.g., ZrCl4) from the supercell structure.
        Continues checking until a tetrahedral structure is found or maximum retries are reached.

        Parameters:
        - center_atom_symbol: Symbol of the central atom (e.g., 'Zr').
        - bonded_symbol: Symbol of atoms bonded to the center atom (e.g., 'Cl' for ZrCl4).
        - bonding_threshold: Distance threshold for bonded atoms.
        - exact_bonded_atoms: Exact number of bonded atoms expected in the molecule.
        - max_retries: Maximum number of attempts to find a tetrahedral structure.

        Returns:
        - An ASE Atoms object representing one unit molecule or None if not found.
        """
        for attempt in range(max_retries):
            # Create a supercell
            supercell_3x3x3 = self.make_supercell()

            for central_atom in supercell_3x3x3:
                if central_atom.symbol != center_atom_symbol:
                    continue

                # Find bonded atoms within the threshold distance
                bonded_atoms = [central_atom]
                for atom in supercell_3x3x3:
                    if atom.symbol == bonded_symbol:
                        distance = np.linalg.norm(atom.position - central_atom.position)
                        if distance <= bonding_threshold:
                            bonded_atoms.append(atom)

                # Check if the number of bonded atoms matches the expected exact number
                num_bonded_atoms = len(bonded_atoms) - 1  # Exclude the central atom
                if num_bonded_atoms == exact_bonded_atoms:
                    isolated_molecule = Atoms(bonded_atoms, cell=supercell_3x3x3.get_cell(), pbc=False)

                    # Check if the structure is tetrahedral
                    if self.is_tetrahedral(center_atom_symbol=center_atom_symbol, bonded_symbol=bonded_symbol,
                                            bonding_threshold=bonding_threshold, exact_bonded_atoms=exact_bonded_atoms):
                        print(f"Isolated tetrahedral molecule: {len(isolated_molecule)} atoms, "
                              f"Composition: {dict(Counter(atom.symbol for atom in bonded_atoms))}")
                        return isolated_molecule

            print("No tetrahedral structure found in this supercell. Retrying...")
        
        print("Maximum retries reached. No tetrahedral structure found.")
        return None

    def display_expanded_isolated_molecule(self, center_atom_symbol='Zr', bonded_symbol='Cl', bonding_threshold=2.7,
                                            exact_bonded_atoms=4):
        """
        Display the isolated molecule in an expanded manner.

        Parameters:
        - center_atom_symbol: Symbol of the central atom (e.g., 'Zr').
        - bonded_symbol: Symbol of atoms bonded to the center atom (e.g., 'Cl' for ZrCl4).
        - bonding_threshold: Distance threshold for bonded atoms.
        - exact_bonded_atoms: Exact number of bonded atoms expected in the molecule.
        """
        isolated_molecule = self.isolate_unit_molecule(center_atom_symbol, bonded_symbol, bonding_threshold, exact_bonded_atoms)

        if isolated_molecule is not None:
            print(f"Displaying expanded isolated molecule: {len(isolated_molecule)} atoms.")
            view = nv.show_ase(isolated_molecule)
            return view
        else:
            print("No isolated molecule found to display.")

    def is_tetrahedral(self, center_atom_symbol='Zr', bonded_symbol='Cl', bonding_threshold=2.7, exact_bonded_atoms=4):
        """
        Check if the isolated molecule forms a tetrahedral structure.

        Parameters:
        - center_atom_symbol: Symbol of the central atom (e.g., 'Zr').
        - bonded_symbol: Symbol of atoms bonded to the center atom (e.g., 'Cl' for ZrCl4).
        - bonding_threshold: Distance threshold for bonded atoms.
        - exact_bonded_atoms: Exact number of bonded atoms expected in the molecule.

        Returns:
        - True if the structure is tetrahedral, False otherwise.
        """
        isolated_molecule = self.isolate_unit_molecule(center_atom_symbol, bonded_symbol, bonding_threshold, exact_bonded_atoms)

        if isolated_molecule is None or len(isolated_molecule) != exact_bonded_atoms + 1:
            print("Not enough atoms to determine tetrahedral structure.")
            return False

        # Calculate the coordinates of the bonded atoms
        central_atom_pos = isolated_molecule[0].position
        bonded_atoms_pos = [atom.position for atom in isolated_molecule[1:]]

        # Calculate distances from the central atom to the bonded atoms
        distances = [np.linalg.norm(bonded_atom - central_atom_pos) for bonded_atom in bonded_atoms_pos]

        # Check if distances are approximately equal (within a tolerance)
        if all(abs(dist - distances[0]) < 0.4 for dist in distances):
            print(f"The molecule is tetrahedral with central atom: {center_atom_symbol}")
            return True
        else:
            print(f"The molecule is not tetrahedral with central atom: {center_atom_symbol}")
            return False

