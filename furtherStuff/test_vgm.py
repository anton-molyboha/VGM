"""
This file tests and gives usage-examples of the VGM-package. 
Execute it e.g. via 'sage -python test_vgm.py'.
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

Import the module and construct vascular graphs in various ways:
>>> import vgm  
>>> import numpy as np

>>> G = vgm.read_amira_spatialGraph_v2('spatialGraph.am')
Vertices: 222
Edges: 165
Points: 9354
reading and adding vertices
reading connectivity information
reading number of edge points
reading point coordinates
10.0%
20.0%
30.0%
40.0%
50.0%
60.0%
70.0%
80.0%
90.0%
100.0%
reading radii
adding edges
>>> G.vcount()
222
>>> G.ecount()
165


Add the edge property conductance, as well as pressure boundary conditions at
specific vertices. Then construct and solve a linear system of equations to 
obtain pressure (vertex preperty) and flow (edge property):
>>> vgm.add_conductance(G,'a')
>>> vgm.add_pBCs(G,'a',[21])
>>> vgm.add_pBCs(G,'v',[42])
>>> LS = vgm.LinearSystem(G)
>>> LS.solve(method='direct')
>>> 'pressure' in G.vs.attribute_names()
True
>>> 'flow' in G.es.attribute_names()
True


Substances can be added by name. Here, we add 'CO2'. Substance (concentration)
boundary conditions are set at specific nodes. Then we construct an Advection
object and advect the substance for a given time:
>>> G.add_substance('CO2')
>>> G.add_sBCs('CO2', [21], [1.0])
>>> vgm.add_geometric_edge_properties(G)
>>> A = vgm.Advection(G,'CO2',1.0)
dt: 136.332608 ms
>>> A.advect(time=10)

Diffusion is implemented analogously to advection. Here, we demonstrate how to
diffuse for a given number of steps (as opposed to a specific time), update the
substance concentration of the VascularGraph each timestep and write the 
results in a Paraview readable file-format. 
>>> D = vgm.Diffusion(G, 'CO2', 1.0)
dt: 0.098365
>>> outfile = 'timeSeries.pvd'
>>> filenames = []
>>> times = []
>>> filenames.append('c0.vtp')
>>> times.append(D.time)
>>> D.set_concentration()
>>> vgm.write_vtp(G,filenames[0],False)
>>> for step in xrange(1,21):
...    D.diffuse(steps=10)
...    filenames.append('c' + str(step) + '.vtp')
...    times.append(D.time)
...    D.set_concentration()
...    vgm.write_vtp(G,filenames[step],False)
>>> vgm.write_pvd_time_series(outfile,filenames,times)

This demonstrates the usage of the fitting routines. Data are generated that 
are sin(x) + a random component. A sine function is fit to the data:
>>> x = np.linspace(0,np.pi)
>>> y = 10 * np.sin(x)
>>> y = y + (np.random.random(x.shape[0]) * 2.0 - 1.0)
>>> fitfnc, p0 = vgm.sine()
>>> results = vgm.fit(fitfnc, p0, [x,y])

"""

if __name__ == '__main__':
    import doctest
    doctest.testmod()
