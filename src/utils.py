import nglview as nv
from ase.io import write

def visualize_atoms(qc_atoms, ecp_atoms, pc_atoms):
    """
    Visualize different layers of atoms using NGLView.

    Parameters:
    - qc_atoms (ase.Atoms): Atoms object for the QC layer.
    - ecp_atoms (ase.Atoms): Atoms object for the ECP layer.
    - pc_atoms (ase.Atoms): Atoms object for the PC layer.
    """


    # Create separate views for each structure using ASE atoms
    view_qc = nv.show_ase(qc_atoms)  # For QC
    view_ecp = nv.show_ase(ecp_atoms)  # For ECP
    view_pc = nv.show_ase(pc_atoms)   # For PC

    # Add representations for each structure as balls
    view_qc.add_representation('ball', component=0)  # For QC
    view_ecp.add_representation('ball', component=0)  # For ECP
    view_pc.add_representation('ball', component=0)  # For PC

    # Display views separately
    return view_qc, view_ecp, view_pc
