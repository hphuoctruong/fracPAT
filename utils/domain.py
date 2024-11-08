import fenics as fn
import numpy as np

def comp_dS(mesh, subdomains):
    int_boundary = fn.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    tdim = mesh.topology().dim()
    mesh.init(tdim - 1, tdim)
    facet_to_cell = mesh.topology()(tdim - 1, tdim)
    num_facets = facet_to_cell.size()
    domain_values = subdomains.array()
    facet_values = int_boundary.array()
    for facet in range(num_facets):
        # Check if interior:
        cells = facet_to_cell(facet)
        if len(cells == 2):
            # Check if facet is on the boundary between two cells with different markers:
            values = np.unique(domain_values[cells])
            if len(values) == 2:
                facet_values[facet] = 1
        else:
            continue
    ds = fn.Measure("dS", domain=mesh, subdomain_data=int_boundary)
    return ds


