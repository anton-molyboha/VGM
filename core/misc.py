from __future__ import division                   

import igraph as ig
import numpy as np
from pylab import flatten
import scipy as sp
from scipy import (array, arccos, argmin, concatenate, dot, ones, mean, pi, 
                   shape, unique, weave)
from scipy.linalg import norm
from scipy.spatial import kdtree
#THOSE lines produce error sometimes switch them off
from sympy.solvers import solve
from sympy import Symbol

from linearSystem import LinearSystem
from physiology import Physiology 

__all__ = ['add_geometric_edge_properties', 'add_fluiddynamical_properties',
           'vertices_from_coordinates', 'add_kind_and_conductance', 
           'add_conductance', 'edge_property_vs_depth', 'update_lengths',
           'update_length', 'update_volume', 'update_depth']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def add_geometric_edge_properties(G):
    """ Adds angle to cortical surface (in degrees), cortical depth, volume and
    cross section to each edge in the graph.
    INPUT: G:  Vascular graph in iGraph format.
    OUTPUT: None - the vascular graph G is modified in place.
    """
           
    depth = []
    angle = []
    crossSection = [] 
    volume = []

    ez = array([0,0,1])
    
    for edge in G.es:         
        
        a = G.vs[edge.source]['r']
        b = G.vs[edge.target]['r']
        v = a-b    
        depth.append((a[2]+b[2])/2.0)
        
        theta=arccos(dot(v,ez)/norm(v))/2/pi*360
        if theta > 90:
            theta = 180-theta
        angle.append(theta)
    
        crossSection.append(np.pi * edge['diameter']**2. / 4.)
        volume.append(crossSection[-1] * edge['length'])
    
    
    G.es['depth'] = depth
    G.es['angle'] = angle
    G.es['volume'] = volume
    G.es['crossSection'] = crossSection 
           

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def add_fluiddynamical_properties(G):
    """Adds transit-time and velocity to each edge in the graph.
    INPUT: G:  Vascular graph in iGraph format.
    OUTPUT: None - the vascular graph G is modified in place.
    WARNING: The edge property 'flow' is expected to be assigned and is 
             interpreted as volume flow.
    """

    transitTime = []
    velocity = []
    
    ez = array([0,0,1])
    for edge in G.es: 
        points = edge['points']
        diameters = edge['diameters']
        flow = edge['flow']
        volume = edge['volume']
        tmpVelocity = []
        for i in xrange(len(points)-1):
            l = norm(points[i+1] - points[i])
            A = pi * mean([diameters[i+1],diameters[i]])**2.0 / 4.0
            tmpVelocity.append(A/flow)
        transitTime.append(volume / flow)
        velocity.append(mean(tmpVelocity))

    G.es['transitTime'] = transitTime
    G.es['velocity'] = velocity
           

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def vertices_from_coordinates(G, coordinates, diameter_ll=0.0, 
                              isEndpoint=False):
    """Given a list of x,y,z coordinates, locate the most closely matching set
    of verties in a vascular graph of iGraph format.
    INPUT: G:  Vascular graph in iGraph format. 
           coordinates: List of lists specifying the coordinates (i.e.: 
                        [[x1,y1,z1],[x2,y2,z2],...]).
           diameter_ll: (Optional) lower limit of edge diameter, to select only
                        those vertices bordering a sufficiently large diameter
                        edge. Default is 0.0, i.e. all vertices are considered.
           isEndpoint: Boolean whether or not the vertex searched for is 
                       required to be an endpoint. Default is 'False'.
    OUTPUT: vertex_indices: Array of vertex indices that represent the best 
                            matches.
            distances: Array of distances that the best matching vertices are 
                       separated from the supplied coordinates. Units match 
                       those of the graph vertex coordinates.
    """
    
    # Select vertex indices based on diameter of adjacent edges:
    si = unique(flatten([G.es[x].tuple for x in 
         G.es(diameter_ge=diameter_ll).indices])).tolist()
    # Optionally filter for end-points:
    if isEndpoint:
        si = [i for i in si if G.degree(i) == 1]
    # Construct k-dimensional seach-tree:
    kdt = kdtree.KDTree(G.vs[si]['r'], leafsize=10)
    search_result = kdt.query(coordinates)
    sr_v = np.ravel([search_result[1]]).tolist()
    vertex_indices = [si[x] for x in sr_v]
    distances = np.ravel([search_result[0]]).tolist()    

    return vertex_indices, distances 
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def add_kind_and_conductance(G,aVertices,vVertices, dThreshold):
    """Adds vessel kind as well as conductance to the graph. The vessel kind is
    assigned based on a connected component analysis and vertices of known 
    kind. The vessel conductance is computed from fitted literature values. At 
    capillary level, a distinction between 'artery' and 'vein' is difficult to 
    make. Moreover, the relative apparent viscosity does not differ 
    dramatically. Therefore, all capillaries (as well as undefined vessels) are
    set to arterial conductance.    
    INPUT: G: VascularGraph.
           aVertices: Vertices that belong to arterial trees.
           vVertices: Vertices that belong to venous trees.
           dThreshold: The diameter threshold below equal which vessels are 
                       considered as capillaries. 
    OUTPUT: None - G is modified in-place.                   
    """                          
    
    # Arteries and veins:
    for kind, indices in zip(['a','v'],[aVertices, vVertices]):
        for vIndex in indices:
            treeVertices = G.get_tree_subcomponent(vIndex, dThreshold)
            edgeIndices = G.es(G.get_vertex_edges(treeVertices), 
                               diameter_gt=dThreshold).indices
            G.es(edgeIndices)['kind'] = [kind for e in edgeIndices]                   
            add_conductance(G,kind,invivo,edgeIndices)

    # Capillaries:
    capillaryIndices = G.es(diameter_le=dThreshold).indices
    G.es(capillaryIndices)['kind'] = ['c' for c in capillaryIndices]            
    add_conductance(G,'a',invivo,capillaryIndices)
    
    # Not assigned:
    notAssigned = []
    for e in G.es:
        if e['kind'] is None:
            notAssigned.append(e.index)
    G.es(notAssigned)['kind'] = ['n' for c in notAssigned]            
    add_conductance(G,'a',invivo,notAssigned)            

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def add_conductance(G,kind,invivo,edges=None):
    """Adds conductance values to the edges of the graph (consult the relevant
    functions in the physiology module for more information.
    INPUT: G: Vascular graph in iGraph format.
           kind: The vessel kind. This can be either 'a' for artery or 'v' for
                 vein.
           invivo: Boolean, whether the physiological blood characteristics 
                   are calculated using the invivo (=True) or invitro (=False)
                   equations
           edges: (Optional.) The indices of the edges to be given a 
                  conductance value. If no indices are supplied, all edges are 
                  considered.
    """
    P = Physiology(G['defaultUnits'])
    if edges is None:
        edgelist = G.es
    else:
        edgelist = G.es(edges)
    #for e in edgelist:
        #print('')
        #print(e['diameter'])
        #print(e['length'])
        #print(P.dynamic_blood_viscosity(e['diameter'],invivo,kind))
    G.es(edgelist.indices)['conductance'] = \
                                   [P.conductance(e['diameter'],e['length'],
                                       P.dynamic_blood_viscosity(e['diameter'],
                                                                 invivo,kind))
                                    for e in edgelist]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def edge_property_vs_depth(G,property,intervals,eIndices=None,function=None):
    """Generic function to compile and optionally process edge information of a
    vascular graph versus the cortical depth.
    INPUT: G: Vascular graph in iGraph format.
           property: Which edge property to operate on.
           intervals: Intervals of cortical depth in which the sample is split.
                      (Expected 
           eIndices: (Optional.) Indices of edges to consider. If not provided,
                     all edges are taken into account.
           function: (Optional.) Function which to perform on the compiled data
                     of each interval.
    OUTPUT: The compiled (and possibly processed) information as a list (one 
            entry per interval).
    """
    
    intervals[-1] = (intervals[-1][0], intervals[-1][1] + sp.finfo(float).eps)
    database = []
    for interval in intervals:
        if eIndices:
            data = G.es(eIndices,depth_ge=interval[0], 
                        depth_lt=interval[1])[property]
        else:                
            data = G.es(depth_ge=interval[0], 
                        depth_lt=interval[1])[property]
        if function:
            database.append(function(data))
        else:
            database.append(data)
    return database   


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def update_lengths(G):
    """Updates the lengths property of all edges in the VascularGraph (adds it,
    if it does not exist).
    INPUT: G: VascularGraph.
    OUTPUT: None
    """
    lengths = []
    for e in G.es:
        tmpLengths = [np.linalg.norm(e['points'][i] - e['points'][i+1]) 
                      for i in xrange(len(e['points']) - 1)]
        tmpLengths.insert(0, 0.0)
        tmpLengths.append(0.0)
        lengths.append((np.array(tmpLengths[:-1]) + 
                       np.array(tmpLengths)[1:]) / 2.0)
    G.es['lengths'] = lengths
    
        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def update_length(G):
    """Updates the length property of all edges in the VascularGraph (adds it,
    if it does not exist).
    INPUT: G: VascularGraph.
    OUTPUT: None
    """
    if 'lengths' in G.es.attribute_names():
        G.es['length'] = [sum(e['lengths']) for e in G.es]
    else:
        G.es['length'] = [np.linalg.norm(G.vs[e.source]['r'] - 
                                         G.vs[e.target]['r']) for e in G.es]    


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def update_volume(G):
    """Updates the volume property of all edges in the VascularGraph (adds it,
    if it does not exist).
    INPUT: G: VascularGraph.
    OUTPUT: None
    """
    G.es['volume'] = [np.pi * e['diameter']**2 / 4.0 * e['length'] 
                      for e in G.es]
        

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def update_depth(G):
    """Updates the depth property of all edges in the VascularGraph (adds it,
    if it does not exist).
    INPUT: G: VascularGraph.
    OUTPUT: None
    """
    G.es['depth'] = [np.mean([G.vs[x[0]]['r'][2], G.vs[x[1]]['r'][2]]) 
                     for x in G.get_edgelist()]
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def intersection_plane_line(pP,nP,pL,vL):
    """Computes the intersection of a plane and a line
    INPUT: pPlane: point on plane
           nPlane: normal vector of plane
           pLine: point on line
           vLine: vecotr on line
    OUTPUT: coordinates of intersectionPoint
    """

    plane = lambda x1,x2,x3: (x1-pP[0])*nP[0]+(x2-pP[1])*nP[1]+(x3-pP[2])*nP[2]

    iP=Symbol('iP')

    #intersection 
    iP=solve(plane(pL[0]+iP*vL[0],pL[1]+iP*vL[1],pL[2]+iP*vL[2]),iP)

    #Compute intersection point
    Point = lambda iP: [pL[0]+iP*vL[0],pL[1]+iP*vL[1],pL[2]+iP*vL[2]]

    if iP != []:
        coordsPoint = Point(iP[0])
    else:
        coordsPoint = []

    return coordsPoint
