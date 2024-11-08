from utils.functions import *


class SubDomain:
    """ Class storing one subdomain of the mesh."""

    def __init__(self, mesh, V):
        self.mesh = mesh
        self.V = V
        self.subdomains = fn.MeshFunction("size_t", mesh, 2, mesh.domains())
        self.boundaries = fn.MeshFunction('size_t', mesh, 1, mesh.domains())

        for f in fn.facets(mesh):
            domains = []
            for c in fn.cells(f):
                domains.append(self.subdomains[c])
            domains = list(set(domains))
            if len(domains) > 1:
                self.boundaries[f] = 1

        # Assemble the boundary matrix:
        self.B_matrix = fn.Matrix(fn.PETScMatrix())

    def assemble_boundary(self):
        dS = fn.dS(subdomain_data=self.boundaries)
        u_trial, u_test = fn.TrialFunction(self.V), fn.TestFunction(self.V)
        fn.assemble(fn.inner(u_trial('+'), u_test('+')) * dS(1), tensor=self.B_matrix)
        u_temp = init_vector(self.B_matrix, 0)
        u_temp[:] = 1.
        Bu_temp = self.B_matrix * u_temp
        self.targets = np.nonzero(Bu_temp[:])[0]

    def obs_boundary(self, x, obs_times=None):
        # Observe x on the boundary:
        if obs_times is None:
            x_obs = init_vector(self.B_matrix, 0)
            x_obs[self.targets] = x[self.targets]
        return x_obs


