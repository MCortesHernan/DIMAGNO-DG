import gmsh
import meshio
import numpy as np

class MeshCreator():
    def __init__(self, coordinates, scalar, name):
        '''
        This class contains method to create a mesh in msh format via gmsh mesh_create and to
        transform it info dolfin-readable xdmf format via meshio.

        It takes the following arguments:
            *coordinates: np array with coordinates of the vertices that bound the domain.
            Expected shape must be [#points, dimension].
            If theres a special point in the boundaries that should have a node 
            (eg: separation of inlet/outlet in the same lateral) it must be
            included here as well. 

            *scalar: R^n -> R Dolfin function giving the typical mesh size required in the domain.

            *name = string with save name for msh and xdmf file (without extension).

        '''
        self.coordinates = coordinates
        self.scalar = scalar
        self.N, self.dim = np.shape(self.coordinates)
        self.name = name


    def mesh_create(self):

        gmsh.initialize()

        x_coords = self.coordinates[:,0]
        z_coords = self.coordinates[:,1]

        points = []
        lines = []
        
        for i in range(self.N):
            points.append(gmsh.model.geo.addPoint(x_coords[i], z_coords[i], 0.0))
        
        for i in range(-1, self.N-1):
            lines.append(gmsh.model.geo.addLine(points[i], points[i+1]))

        loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([loop])

        inlet_tag, outlet_tag, wall_tag, domain_tag = 1, 2, 3, 4

        gmsh.model.addPhysicalGroup(1, [lines[-2], lines[-1]], inlet_tag)
        gmsh.model.setPhysicalName(1, inlet_tag, 'Inlet')

        gmsh.model.addPhysicalGroup(1, [lines[0]], wall_tag)
        gmsh.model.setPhysicalName(1, wall_tag, 'Wall')

        gmsh.model.addPhysicalGroup(1, lines[1:-2], outlet_tag)
        gmsh.model.setPhysicalName(1, outlet_tag, 'Outlets')

        gmsh.model.addPhysicalGroup(2, [surface], domain_tag)
        gmsh.model.setPhysicalName(2, domain_tag, 'Domain')

        gmsh.model.geo.synchronize()

        def mesh_size(entity_dim, entity_tag, x, y, z, lc):
            return self.scalar(x,y) 

        gmsh.model.mesh.setSizeCallback(mesh_size)


        gmsh.model.mesh.generate()
        gmsh.write(f'{self.name}.msh')

        gmsh.finalize()

        return None

    def mesh_convert(self, cell_type='triangle', prune_z= True):

        mesh_from_file = meshio.read(f'{self.name}.msh')
        cells = mesh_from_file.get_cells_type(cell_type)
        cell_data = mesh_from_file.get_cell_data("gmsh:physical", cell_type)
        points = mesh_from_file.points[:,:2] if prune_z else mesh_from_file.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        
        meshio.write(f'{self.name}.xdmf', out_mesh)
        
        return None
