'''Convert mesh from gmesh (.msh) format to fenics readable (.xdmf)'''

import meshio

def convert_mesh(path, out_file_name, cell_type='triangle', prune_z= True):

    mesh_from_file = meshio.read(path)
    cells = mesh_from_file.get_cells_type(cell_type)
    cell_data = mesh_from_file.get_cell_data("gmsh:physical", cell_type)
    points = mesh_from_file.points[:,:2] if prune_z else mesh_from_file.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    
    meshio.write(out_file_name, out_mesh)
    