import nglview as nv
from ase.io import write

def visualize_atoms(qc_atoms, ecp_atoms, pc_atoms, combined=False):
    """
    Visualize different layers of atoms using NGLView.

    Parameters:
    - qc_atoms (ase.Atoms): Atoms object for the QC layer.
    - ecp_atoms (ase.Atoms): Atoms object for the ECP layer.
    - pc_atoms (ase.Atoms): Atoms object for the PC layer.
    - combined (bool): If True, visualizes all layers in a single view. 
                       If False, visualizes each layer separately.
    
    Returns:
    - combined_view (nglview.NGLWidget): A single combined view of all layers (if combined=True).
    - view_qc, view_ecp, view_pc (nglview.NGLWidget): Separate views of each layer (if combined=False).
    """

    if combined:
        # Create a single NGLView instance for all layers
        combined_view = nv.NGLWidget()

        # Add the QC layer as the first component
        qc_component = combined_view.add_component(nv.ASEStructure(qc_atoms))
        combined_view.add_representation('ball', selection=qc_component)

        # Add the ECP layer as the second component
        ecp_component = combined_view.add_component(nv.ASEStructure(ecp_atoms))
        combined_view.add_representation('ball', selection=ecp_component)

        # Add the PC layer as the third component
        pc_component = combined_view.add_component(nv.ASEStructure(pc_atoms))
        combined_view.add_representation('ball', selection=pc_component)

        # Center the view on the atoms
        combined_view.center()

        return combined_view

    else:
        # Create separate views for each layer
        view_qc = nv.show_ase(qc_atoms)  # QC layer
        view_ecp = nv.show_ase(ecp_atoms)  # ECP layer
        view_pc = nv.show_ase(pc_atoms)   # PC layer

        # Add ball representation for each layer
        view_qc.add_representation('ball')
        view_ecp.add_representation('ball')
        view_pc.add_representation('ball')

        # Return separate views
        return view_qc, view_ecp, view_pc
