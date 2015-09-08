"""This module implements red blood cell transport in vascular networks 
discretely, i.e. resolving all RBCs. As would be expected, the computational 
expense increases with network size and timesteps can become very small. An
srXTM sample of ~20000 nodes will take about 1/2h to evolve 1ms at Ht0==0.5.
A performance analysis revealed the worst bottlenecks, which are:
_plot_rbc()
_update_blocked_edges_and_timestep()
Smaller CPU-time eaters are:
_update_flow_and_velocity()
_update_flow_sign()
_propagate_rbc()
"""
from __future__ import division

import numpy as np
from sys import stdout

from copy import deepcopy
from pyamg import ruge_stuben_solver, smoothed_aggregation_solver
import pyamg
from scipy import finfo, ones, zeros
from scipy.sparse import lil_matrix, linalg
from physiology import Physiology
import units
import g_output
import vascularGraph
import pdb
import time as ttime
import vgm

__all__ = ['LinearSystemHtd']
log = vgm.LogDispatcher.create_logger(__name__)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class LinearSystemHtd(object):
    """Implements and extends the discrete red blood cell transport as proposed
    by Obrist and coworkers (2010). The pressure boundary condition should be given
    in mmHg.
    """
    #@profile
    def __init__(self, G, invivo=True, dThreshold=10.0, init=True,**kwargs):
        """Initializes a LinearSystemHtd instance.
        INPUT: G: Vascular graph in iGraph format.
               invivo: Boolean, whether the physiological blood characteristics 
                       are calculated using the invivo (=True) or invitro (=False)
                       equations
               dThreshold: Diameter threshold below which vessels are
                           considered capillaries, to which the reordering
                           algorithm can be applied.
               init: Assign initial conditions for RBCs (True) or keep old positions to
                        continue a simulation (False)
               **kwargs:
               ht0: The initial hematocrit in the capillary bed. If G already 
                    contains the relevant properties from an earlier simulation
                    (i.e. rRBC edge property), ht0 can be set to 'current'.
                    This will make use of the current RBC distribution as the
                    initial hematocrit
               hd0: The initial hematocrit in the capillary bed is calculated by the given
                    initial discharge hematocrit. If G already contains the relevant 
                    properties from an earlier simulation (i.e. rRBC edge property), hd0 
                    can be set to 'current'. This will make use of the current RBC 
                    distribution as the initial hematocrit
               plasmaViscosity: The dynamic plasma viscosity. If not provided,
                                a literature value will be used.
               pTracking:Boolean whether RBC tracking output should be written
        OUTPUT: None, however the following items are created:
                self.A: Matrix A of the linear system, holding the conductance
                        information.
                self.b: Vector b of the linear system, holding the boundary
                        conditions.
        """
        self._G = G
        self._P = Physiology(G['defaultUnits'])
        self._dThreshold = dThreshold
        self._invivo=invivo
        nVertices = G.vcount()
        self._b = zeros(nVertices)
        self._x = zeros(nVertices)
        self._A = lil_matrix((nVertices,nVertices),dtype=float)
        self._eps = finfo(float).eps * 1e4
        self._blockedEdges = []
        self._tPlot = 0.0
        self._tSample = 0.0
        self._filenamelist = []
        self._timelist = []
        self._filenamelistAvg = []
	self._timelistAvg = []
        self._sampledict = {} 
        self._RBCdict= {}
        self._RBCindexCurrent=0
	self._init=init
        self._scaleToDef=vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])

        # Assure that both pBC and rBC edge properties are present:
        for key in ['pBC', 'rBC']:
            if not G.vs[0].attributes().has_key(key):
                G.vs[0][key] = None

        #Particle Tracking
        if kwargs.has_key('pTracking'):
            self._pTracking=kwargs['pTracking']
        else:
            self._pTracking=0
        pTracking = self._pTracking

        # Set initial pressure and flow to zero:
	if init:
          G.vs['pressure']=zeros(nVertices)                                                                                    
          G.es['flow']=zeros(G.ecount())    

	#Calculate total network Volume
        G['V']=sum([0.25*np.pi*e['diameter']**2*e['length'] for e in G.es]) 

        # Set initial bifurcation vertex, in-edge and timestep:
        self._vi = None
        self._eiIn = None
        self._dt = -1
        
        # Compute the edge-specific minimal RBC distance:
        vrbc = self._P.rbc_volume()
        G.es['minDist'] = [vrbc / (np.pi * e['diameter']**2 / 4) for e in G.es]
        G.es['nMax'] = [e['length']/ e['minDist'] for e in G.es] 

        # Group vertices in capillary, noncapillary, and interface:
        self._interfaceNoncapNeighborsVI = {}
        self._interfaceNoncapAdjacentEI = {}
        self._capVerticesI = []
        self._noncapVerticesI = []
        self._interfaceVerticesI = []
        for vi in xrange(G.vcount()):
            adjEdgesI = G.adjacent(vi)
            adjNoncapEdges = G.es(adjEdgesI, diameter_gt=dThreshold)
            if len(adjNoncapEdges) > 0:
                if len(adjNoncapEdges) == len(adjEdgesI):
                    self._noncapVerticesI.append(vi)
                else:
                    self._interfaceVerticesI.append(vi)
                    self._interfaceNoncapNeighborsVI[vi] = [e.source if e.source != vi else e.target for e in adjNoncapEdges]
                    self._interfaceNoncapAdjacentEI[vi] = adjNoncapEdges.indices
            elif len(adjEdgesI) > 0:
                self._capVerticesI.append(vi)
        G.vs[self._capVerticesI]['isCap'] = [True for v in self._capVerticesI]
        G.vs[self._noncapVerticesI]['isCap'] = [False for v in self._noncapVerticesI]

        # Arterial-side inflow:
        if 'htdBC' in G.es.attribute_names():
           G.es['httBC']=[e['htdBC'] if e['htdBC'] == None else \
                self._P.discharge_to_tube_hematocrit(e['htdBC'],e['diameter'],invivo) for e in G.es()]
        if not 'httBC' in G.es.attribute_names():
            for vi in G['av']:
                for ei in G.adjacent(vi):
                    G.es[ei]['httBC'] = self._P.tube_hematocrit(
                                            G.es[ei]['diameter'], 'a')
        httBC_edges = G.es(httBC_ne=None).indices
        self._inflowTracker = dict(zip(httBC_edges, [[0., False] for x in httBC_edges]))

        # Assign initial RBC positions:
	if init:
            if kwargs.has_key('hd0'):
                hd0=kwargs['hd0']
                if hd0 == 'current':
                    ht0=hd0
                else:
                    ht0='dummy'
            if kwargs.has_key('ht0'):
                ht0=kwargs['ht0']
            if ht0 != 'current':
                for e in G.es:
                    lrbc = e['minDist']
                    Nmax = max(int(np.floor(e['nMax'])), 1)
                    if e['httBC'] is not None:
                        N = int(np.round(e['httBC'] * Nmax))
                    else:
                        if kwargs.has_key('hd0'):
                            ht0=self._P.discharge_to_tube_hematocrit(hd0,e['diameter'],invivo)
                        N = int(np.round(ht0 * Nmax))
                    indices = sorted(np.random.permutation(Nmax)[:N])
                    e['rRBC'] = np.array(indices) * lrbc + lrbc / 2.0

        if pTracking:
            for e in G.es:
	        e['RBCindex'] = np.array([])
                    
        if kwargs.has_key('plasmaViscosity'):
            self._muPlasma = kwargs['plasmaViscosity']
        else:
            self._muPlasma = self._P.dynamic_plasma_viscosity()

        # Compute nominal and specific resistance:
        self._update_nominal_and_specific_resistance()

        # Compute the current tube hematocrit from the RBC positions:
        self._update_tube_hematocrit()
        
        # This initializes the full LS. Later, only relevant parts of
        # the LS need to be changed at any timestep. Also, RBCs are
        # removed from no-flow edges to avoid wasting computational
        # time on non-functional vascular branches / fragments:

        #Convert 'pBC' ['mmHG'] to default Units
        G.vs['pBC']=[v*self._scaleToDef if v != None else None for v in G.vs['pBC']]
        self._update_eff_resistance_and_LS(None, None)
        self._solve('direct')
        self._G.vs['pressure'] = deepcopy(self._x)
        self._update_flow_and_velocity()
        #Convert defaultUnits to 'pBC' ['mmHG']                                                                             
        G.vs['pBC']=[v/self._scaleToDef if v != None else None for v in G.vs['pBC']]                 
	#Calculate an estimated network turnover time (based on conditions at the beginning)
        flowsum=0
	for vi in G['av']:
            for ei in G.adjacent(vi):
                flowsum=flowsum+G.es['flow'][ei]
        G['Ttau']=G['V']/flowsum
        stdout.write("\rEstimated network turnover time Ttau=%f        \n" % G['Ttau'])

        for e in self._G.es(flow_le=self._eps*1e6):
            e['rRBC'] = []
            
    #--------------------------------------------------------------------------

    def _update_nominal_and_specific_resistance(self, esequence=None):
        """Updates the nominal and specific resistance of a given edge 
        sequence.
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge properties 'resistance' and 'specificResistance'
                are updated (or created).
        """
        G = self._G
        muPlasma=self._muPlasma
        pi=np.pi  

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)
        es['specificResistance'] = [128 * muPlasma / (pi * d**4)
                                        for d in es['diameter']]

        es['resistance'] = [l * sr for l, sr in zip(es['length'], 
                                                es['specificResistance'])]

	self._G = G

    #--------------------------------------------------------------------------

    def _update_minDist_and_nMax(self, esequence=None):
        """Updates the length of the RBCs for each edge and the maximal Number
		of RBCs for each edge
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge properties 'nMax' and 'minDist'
                are updated (or created).
        """
        G = self._G

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)
        # Compute the edge-specific minimal RBC distance:
        vrbc = self._P.rbc_volume()
        G.es['nMax'] = [np.pi * e['diameter']**2 / 4 * e['length'] / vrbc
                        for e in G.es]
        G.es['minDist'] = [e['length'] / e['nMax'] for e in G.es]

	self._G=G

    #--------------------------------------------------------------------------

    def _update_tube_hematocrit(self, esequence=None):
        """Updates the tube hematocrit of a given edge sequence.
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge property 'htt' is updated (or created).
        """
        G = self._G

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)
        es['htt'] = [len(e['rRBC']) * e['minDist'] / e['length'] for e in es]

	self._G=G

    #--------------------------------------------------------------------------

    def _update_local_pressure_gradient(self):
        """Updates the local pressure gradient at all vertices.
        INPUT: None
        OUTPUT: None, the edge property 'lpg' is updated (or created, if it did
                not exist previously)
        """
        G = self._G
#        G.es['lpg'] = np.array(G.es['specificResistance']) * \
#                      np.array(G.es['flow']) * np.array(G.es['resistance']) / \
#                      np.array(G.es['effResistance'])
        G.es['lpg'] = np.array(G.es['specificResistance']) * \
                      np.array(G.es['flow'])

        self._G=G
    #--------------------------------------------------------------------------

    def _update_interface_vertices(self):
        """(Re-)assigns each interface vertex to either the capillary or non-
        capillary group, depending on whether ist inflow is exclusively from
        capillaries or not.
        """
        G = self._G
        dThreshold = self._dThreshold

        for v in self._interfaceVerticesI:
            p = G.vs[v]['pressure']
            G.vs[v]['isCap'] = True
            for n in self._interfaceNoncapNeighborsVI[v]:
                if G.vs[n]['pressure'] > p:
                    G.vs[v]['isCap'] = False
                    break

    #--------------------------------------------------------------------------

    def _update_flow_sign(self):
        """Updates the sign of the flow. The flow is defined as having a
        positive sign if its direction is from edge source to target, negative
        if vice versa and zero otherwise (in case of equal pressures).
        INPUT: None
        OUTPUT: None (the value of the edge property 'sign' will be updated to
                one of [-1, 0, 1])
        """
        G = self._G
        if 'sign' in G.es.attributes(): 
            G.es['sign_old']=G.es['sign']
        G.es['sign'] = [np.sign(G.vs[e.source]['pressure'] -
                                G.vs[e.target]['pressure']) for e in G.es]

    #-------------------------------------------------------------------------
    #@profile
    def _update_out_and_inflows_for_vertices(self):
        """Calculates the in- and outflow edges for vertices at the beginning.
        Afterwards in every single timestep it is check if something changed
        INPUT: None 
        OUTPUT: None, however the following parameters will be updated:
                G.vs['inflowE']: Time until next RBC reaches bifurcation.
                G.vs['outflowE']: Index of edge in which the RBC resides.
        """    
        G=self._G
        #Beginning    
        inEdges=[]
        outEdges=[] 
        if not 'inflowE' in G.vs.attributes():
            for v in G.vs: 
                vI=v.index
                j=0
                outE=[]
                inE=[]
                for nI in G.neighbors(vI):
                    if G.vs[vI]['pressure'] > G.vs[nI]['pressure']:
                        outE.append(G.adjacent(vI)[j])
                    else:
                        inE.append(G.adjacent(vI)[j])
                    j += 1
                inEdges.append(inE)
                outEdges.append(outE)
            G.vs['inflowE']=inEdges
            G.vs['outflowE']=outEdges
        #Every Time Step
        else:
            if G.es['sign']!=G.es['sign_old']:
                sum=np.array(G.es['sign'])+np.array(G.es['sign_old'])
                edgeList=np.where(sum == 0)[0]
                for e in edgeList:
                    edge=G.es[int(e)]
                    for vI in edge.tuple:
                        j=0
                        outE=[]
                        inE=[]
                        for nI in G.neighbors(vI):
                            if G.vs[vI]['pressure'] > G.vs[nI]['pressure']:
                                outE.append(G.adjacent(vI)[j])
                            else:
                                inE.append(G.adjacent(vI)[j])
                            j += 1
                        G.vs[vI]['inflowE']=inE
                        G.vs[vI]['outflowE']=outE

    #--------------------------------------------------------------------------

    def _update_blocked_edges_and_timestep(self):
        """Updates the time taken for the next RBC to reach a bifurcation. The
        index of the edge in which the RBC resides is returned also, as well as
        the index of the vertex at which the bifurcation rule is to be applied.
        INPUT: None
        OUTPUT: None, however the following parameters will be updated:
                self._dt: Time until next RBC reaches bifurcation.
                self._eiIn: Index of edge in which the RBC resides.
                self._vi: Index of vertex at which the bifurcation rule needs
                          to be applied next.
        """
        G = self._G
        eps = self._eps
        vi = self._vi
        eiInOldAtDtZero = -1 if self._dt > 0.0 else self._eiIn        
#        # Time to reach bifurcation:
#        ti_tuple = sorted([(e['rRBC'][0 if e['sign']==-1 else -1] / e['v'], e.index) 
#                           if len(e['rRBC']) > 0 else (1e25, e.index) for e in G.es])
        
        blockedEdges = []
        dtmin = 1e25
        tlist = dict([(i, 1e25) for i in xrange(G.ecount())])
        
        # Update maximum timestep and blocked-status for all edges:
        for eIn in G.es:
            ei = eIn.index
            vi = eIn.target if eIn['sign'] == 1 else eIn.source
            outEdges = G.vs[vi]['outflowE']

            # No flow / no RBC are not time limiting:
            if eIn['sign'] * len(eIn['rRBC']) == 0:
                tlist[ei] = 1e25
                continue
            # Outflow edges are limiting, as RBCs need to be removed:
            elif len(outEdges) == 0:
                if len(eIn['rRBC']) > 0:
                    s = eIn['rRBC'][0] if eIn['sign'] == -1 else \
                        eIn['length'] - eIn['rRBC'][-1]
                    dt = s / eIn['v']
                    tlist[ei] = dt
                continue

            # Note that an in-edge that has only blocked out-edges is not
            # blocked itself, as an RBC can still move freely with the flow.

            distToBifurcation = eIn['rRBC'][0] if eIn['sign'] == -1 \
                                else eIn['length'] - eIn['rRBC'][-1]

            # If leading RBC at bifurcation, check whether it has free passage
            # to one of the out-edges or whether it is constrained by Ht-limits
            # (note that it can also enter a blocked out-edge):
            if distToBifurcation <= 0.0+eps:
                timeLimits = []
                for outEdge in outEdges:
                    e = G.es[outEdge]
                    # If outEdge == inEdge of the previous timestep, and that
                    # timestep was zero - we have a case of flip-flopping which 
                    # needs to be avoided:
                    if outEdge == eiInOldAtDtZero:
                        blockedEdges.append(ei)
                        eIn['free'] = []
                        dt = 1e25
                        tlist[ei] = dt
                        break                    
                    # If outEdge is devoid of RBCs, it does not constrain flow
                    # in inEdge:
                    if len(e['rRBC']) == 0:
                        dt = 0.0
                        tlist[ei] = dt
                        break
                    else:
                        # Distance from vertex vi to first RBC in outEdge:
                        s = e['rRBC'][0] if e['sign'] == 1 else \
                            e['length'] - e['rRBC'][-1]
                        if s < e['minDist'] - eps:
                            timeLimits.append((e['minDist']-s) / e['v'])
                        else:
                            dt = 0.0
                            tlist[ei] = dt
                            break
                # If the RBC causing the blockage cannot be reassigned, compute
                # the time it takes for the first of the free RBCs to reach the
                # 'traffic jam':
                if len(timeLimits) == len(outEdges):
                    blockedEdges.append(ei)
                    e = G.es[ei]
                    lrbc = e['minDist']
                    e['free'] = [] # The default, i.e. all RBCs are blocked
                    rrbc = e['rRBC']
                    if e['sign'] == 1:
                        for i in range(len(rrbc)-1, 0, -1):
                            j = i - 1
                            if rrbc[i] - rrbc[j] > lrbc + eps:
                                timeLimits.append((rrbc[i] - rrbc[j] - lrbc) / e['v'])
                                e['free'] = range(i)
                                break
                    else:
                        for i in range(len(rrbc)-1):
                            j = i + 1
                            if rrbc[j] - rrbc[i] > lrbc + eps:
                                timeLimits.append((rrbc[j] - rrbc[i] - lrbc) / e['v'])
                                e['free'] = range(j, len(rrbc))
                                break

                    dt = min(timeLimits)

                    tlist[ei] = dt
            # If leading RBC is not at bifurcation, determine the time it needs
            # to reach it:
            else:
                dt = (eIn['length'] - eIn['rRBC'][-1]) / eIn['v'] \
                     if eIn['sign'] == 1 else eIn['rRBC'][0] / eIn['v']
                tlist[ei] = dt

        dtmin, dtminEdgeIndex = sorted(zip(tlist.values(), tlist.keys()))[0]
        # Account for numerical inaccuracy:
        dtmin = max(0.0, dtmin)
   
        e = G.es[dtminEdgeIndex]
        if e['sign'] == 1:
            vi = e.target
        else:
            vi = e.source

        self._blockedEdges = blockedEdges
        self._dt = dtmin
        self._eiIn = dtminEdgeIndex
        self._vi = vi

    #--------------------------------------------------------------------------

    def _update_flow_and_velocity(self):
        """Updates the flow and red blood cell velocity in all vessels
        INPUT: None
        OUTPUT: None
        """
        G = self._G
        invivo=self._invivo
        vf = self._P.velocity_factor
        pi=np.pi
        G.es['flow'] = [abs(G.vs[e.source]['pressure'] -                                           
                            G.vs[e.target]['pressure']) /res                        
                            for e,res in zip(G.es,G.es['effResistance'])]

        # RBC velocity is not defined if tube_ht==0, using plasma velocity
        # instead:
        G.es['v'] = [4 * flow * vf(d, invivo, tube_ht=htt) /                  
                    (pi * d**2) if htt > 0 else                                
                    4 * flow / (pi * d**2)                                     
                    for flow,d,htt in zip(G.es['flow'],G.es['diameter'],G.es['htt'])]

    #--------------------------------------------------------------------------

    def _update_eff_resistance_and_LS(self, newGraph=None, vertex=None):
        """Constructs the linear system A x = b where the matrix A contains the
        conductance information of the vascular graph, the vector b specifies
        the boundary conditions and the vector x holds the pressures at the
        vertices (for which the system needs to be solved). x will have the
        same units of [pressure] as the pBC vertices.

        Note that in this approach, A and b contain a mixture of dimensions,
        i.e. A and b have dimensions of [1.0] and [pressure] in the pBC case,
        [conductance] and [conductance*pressure] otherwise, the latter being
        rBCs. This has the advantage that no re-indexing is required as the
        matrices contain all vertices.
        INPUT: newGraph: Vascular graph in iGraph format to replace the
                         previous self.G. (Optional, default=None.)
        OUTPUT: A: Matrix A of the linear system, holding the conductance
                   information.
                b: Vector b of the linear system, holding the boundary
                   conditions.
        """

        #if newGraph is not None:
        #    self._G = newGraph

        G = self._G
        P = self._P
        A = self._A
        b = self._b
        x = self._x
        invivo = self._invivo

        htt2htd = P.tube_to_discharge_hematocrit
        nurel = P.relative_apparent_blood_viscosity

        if vertex is None:
            vertexList = range(G.vcount())
            edgeList = range(G.ecount())
        else:
            vertexList = np.unique(np.concatenate([[vertex],
                         G.neighbors(vertex)])).tolist()
            edgeList = G.adjacent(vertex)

        
        for it in self._inflowTracker.iteritems():
            if it[1][1]:
                edgeList.append(it[0])
                vertexList.extend(G.es[it[0]].tuple)
        edgeList = G.es(edgeList)
        vertexList = G.vs(vertexList)

        dischargeHt = [min(htt2htd(e, d, invivo), 0.99) for e,d in zip(edgeList['htt'],edgeList['diameter'])]
        edgeList['effResistance'] =[ res * nurel(d, dHt,invivo) for res,dHt,d in zip(edgeList['resistance'],dischargeHt,edgeList['diameter'])]

        for vertex in vertexList:
            i = vertex.index
            A.data[i] = []
            A.rows[i] = []
            b[i] = 0.0
            if vertex['pBC'] is not None:
                A[i,i] = 1.0
                b[i] = vertex['pBC']
            else:
                aDummy=0
                k=0
                neighbors=[]
                for edge in G.adjacent(i,'all'):
                    if G.is_loop(edge):
                        continue
                    j=G.neighbors(i)[k]
                    k += 1
                    conductance = 1 / G.es[edge]['effResistance']
                    neighbor = G.vs[j]
                    # +=, -= account for multiedges
                    aDummy += conductance
                    if neighbor['pBC'] is not None:
                        b[i] = b[i] + neighbor['pBC'] * conductance
                    else:
                        if j not in neighbors:
                            A[i,j] = - conductance
                        else:
                            A[i,j] = A[i,j] - conductance
                    neighbors.append(j)
                    if vertex['rBC'] is not None:
                        b[i] += vertex['rBC']
                A[i,i]=aDummy

        self._A = A
        self._b = b
	self._G = G

    #--------------------------------------------------------------------------

    def _propagate_rbc(self):
        """This assigns the current bifurcation-RBC to a new edge and
        propagates all RBCs until the next RBC reaches at a bifurcation.
        INPUT: None
        OUTPUT: None
        """
        G = self._G
        dt = self._dt # Time to propagate RBCs with current velocity.
        eiIn = self._eiIn # Index of in-edge of current bifurcation-RBC
        vi = self._vi # Index of the vertex at which the bifurcation rule needs
                      # to be applied.
        blockedEdges = self._blockedEdges
        eps = self._eps
        RBCindexCurrent = self._RBCindexCurrent
        pTracking = self._pTracking
        newRBCs = []

        # Move all RBCs with their current velocity for a time dt (which is
        # when the first RBC will reach a bifurcation or a blocked RBC in the
        # case of a blocked edge):
        if dt > 0.0:
            for e in G.es:
                ei = e.index
                displacement = e['v'] * dt
                if ei in blockedEdges:
                    if len(e['free']) > 0:
                        e['rRBC'][e['free']] = e['rRBC'][e['free']] + \
                                               displacement * e['sign']
                else:
                    e['rRBC'] = e['rRBC'] + displacement * e['sign']

                if e['httBC'] is not None:
                    rRBC = []
		    RBCindex = []
                    lrbc = e['minDist']
                    htt = e['httBC']
                    length = e['length']
                    inflowTracker = self._inflowTracker[ei]
                    cum_length = inflowTracker[0] + displacement
                    #log.debug('Tracker: %f' % inflowTracker[0])
                    #log.debug('Displacement: %f' % displacement)
                    #log.debug('CumLength: %f' % cum_length)
                    if cum_length >= lrbc:
                        nrbc_max = cum_length / lrbc
                        nrbc_max_floor = np.floor(nrbc_max)
                        nrbc = sum(np.random.rand(nrbc_max_floor)<htt)
                        lrbc_modulo = (nrbc_max - nrbc_max_floor) * lrbc
                        start = lrbc_modulo 
                        stop = cum_length 
                        for i in range(nrbc):
                            pos = np.random.rand() * (stop - ((nrbc-i)*lrbc) - start) + start
                            start = pos + lrbc
                            rRBC.append(pos)
                        rRBC = np.array(rRBC)
                        for i in range(len(rRBC)):
                            RBCindex.append(RBCindexCurrent+1)
                            newRBCs.append(RBCindexCurrent+1)
                            RBCindexCurrent += 1
                        if e['sign'] == 1:
                            e['rRBC'] = np.concatenate([rRBC, e['rRBC']])
                            if pTracking:
			        e['RBCindex'] = np.concatenate([RBCindex, e['RBCindex']])
                        else:
                            e['rRBC'] = np.concatenate([e['rRBC'], length-rRBC[::-1]])
                            if pTracking:
                                e['RBCindex'] = np.concatenate([e['RBCindex'],RBCindex[::-1]])
                        if nrbc > 0:
                            self._update_tube_hematocrit((ei))
                            inflowTracker[1] = True
                        else:
                            inflowTracker[1] = False
                        inflowTracker[0] = lrbc_modulo
                    else:
                        inflowTracker[0] += displacement
        self._RBCindexCurrent = RBCindexCurrent
        self._newRBCs = newRBCs
            #return
        #if eiIn in blockedEdges:
        #    log.critical('Timestep of zero occuring at a blocked in-edge leading to an infinite loop.')
        #    return

        # Assign the current bifurcation-RBC to a new vessel:
        p = G.vs[vi]['pressure']
        outEdges = G.vs[vi]['outflowE']
        outEdge = None
        if len(outEdges) > 0:
            # Choose out-edge preference according to bifurcation rule:
            if G.vs[vi]['isCap']:
                preferenceList = [x[1] for x in
                                  sorted(zip(G.es[outEdges]['flow'],
                                             outEdges), reverse=True)]
            # Choose out-edge preference based on Kirchhoff:
            else:
                preferenceList = []
                remainingOE = deepcopy(outEdges)
                while len(remainingOE) > 1:
                    oe = remainingOE[0]
                    outflow = sum(G.es[remainingOE]['flow'])
                    flowFractions = [e['flow']/outflow
                                     for e in G.es[remainingOE]]
                    intervals = np.cumsum(flowFractions)
                    rand = np.random.rand()
                    selectedEdge = remainingOE[np.nonzero(np.less(rand, intervals))[0][0]]
                    preferenceList.append(selectedEdge)
                    remainingOE.remove(selectedEdge)
                preferenceList.append(remainingOE[0])
            # Assign out-edge based on preference-list and ht-constraints:
            for e in G.es[preferenceList]:
                if len(e['rRBC']) == 0:
                    outEdge = e
                    break
                else:
                    s = e['rRBC'][0] if e.source == vi \
                        else e['length'] - e['rRBC'][-1]
                    if s >= e['minDist'] - eps:
                        outEdge = e
                        break

        # If a designated out-edge exists, add RBC:
        if outEdge is not None:
            e = outEdge
            if e.source == vi:
                e['rRBC'] = np.concatenate([[0.0 + self._eps], e['rRBC']])
		#Move 'RBCindex' value with RBC to new edge
                if pTracking:
                    if len(G.es[eiIn]['RBCindex']) > 0 and (len(G.es[eiIn]['rRBC']) == len(G.es[eiIn]['RBCindex'])):
		        if G.es[eiIn]['sign'] == 1:		
                            e['RBCindex']=np.concatenate((G.es[eiIn]['RBCindex'][-1::],e['RBCindex']))
		        else:
                            e['RBCindex']=np.concatenate((G.es[eiIn]['RBCindex'][0:1],e['RBCindex']))
            else:
		#Move 'RBCindex' value with RBC to new edge
                if pTracking:
                    e['rRBC'] = np.concatenate([e['rRBC'], [e['length'] - self._eps]])
                    if len(G.es[eiIn]['RBCindex']) > 0 and (len(G.es[eiIn]['rRBC']) == len(G.es[eiIn]['RBCindex'])):
		        if G.es[eiIn]['sign'] == 1:
                            e['RBCindex']=np.concatenate((e['RBCindex'],G.es[eiIn]['RBCindex'][-1::]))
		        else:
		            e['RBCindex']=np.concatenate((e['RBCindex'],G.es[eiIn]['RBCindex'][0:1]))	
            self._update_tube_hematocrit((outEdge.index))

        # Remove RBC from mother vessel and save transit time of RBCs
        time = self._tPlot + self._dt
        if len(outEdges) > 0 and outEdge == None:
            blockedEdges.append(eiIn)
        else:
            e = G.es[eiIn]
            if e['sign'] == 1:
                if pTracking:
		    if len(e['RBCindex']) > 0 and (len(e['rRBC']) == len(e['RBCindex'])):
                        e['RBCindex']=e['RBCindex'][:-1]
                e['rRBC'] = e['rRBC'][:-1]
            else:
                if pTracking:
                    if len(e['RBCindex']) > 0 and (len(e['rRBC']) == len(e['RBCindex'])):
                        e['RBCindex']=e['RBCindex'][1:]
                e['rRBC'] = e['rRBC'][1:]
            self._update_tube_hematocrit((eiIn))

        G.es['nRBC'] = [len(e['rRBC']) for e in G.es]

	self._blockedEdges = blockedEdges

    #--------------------------------------------------------------------------
    #@profile
    def evolve(self, time, method, **kwargs):
        """Solves the linear system A x = b using a direct or AMG solver.
        INPUT: time: The duration for which the flow should be evolved. In case of
	 	     Reset in plotPrms or samplePrms = False, time is the duration 
	 	     which is added
               method: Solution-method for solving the linear system. This can
                       be either 'direct' or 'iterative'
               **kwargs
               precision: The accuracy to which the ls is to be solved. If not
                          supplied, machine accuracy will be used. (This only
                          applies to the iterative solver)
               plotPrms: Provides the parameters for plotting the RBC 
                         positions over time. List format with the following
                         content is expected: [start, stop, step, reset].
                         'reset' is a boolean which determines if the current 
                         RBC evolution should be added to the existing history
                         or started anew. In case of Reset=False, start and stop
			 are added to the already elapsed time.
               samplePrms: Provides the parameters for sampling, i.e. writing 
                           a series of data-snapshots to disk for later 
                           analysis. List format with the following content is
                           expected: [start, stop, step, reset]. 'reset' is a
                           boolean which determines if the data samples should
                           be added to the existing database or a new database
                           should be set up. In case of Reset=False, start and stop
                          are added to the already elapsed time.
               SampleDetailed:Boolean whether every step should be samplede(True) or
			      if the sampling is done by the given samplePrms(False)
               pTracking:Boolean whether RBC tracking output should be written
               BackUp: Boolean whether BackUps s
         OUTPUT: None (files are written to disk)
        """
        G=self._G
        tPlot = self._tPlot # deepcopy, since type is float
        tSample = self._tSample # deepcopy, since type is float
        filenamelist = self._filenamelist
        timelist = self._timelist
	filenamelistAvg = self._filenamelistAvg
	timelistAvg = self._timelistAvg
        self._initRBCdict = 1 
        self._newRBCs = []

        if 'init' in kwargs.keys():
            init=kwargs['init']
        else:
            init=self._init

        SampleDetailed=False
        if 'SampleDetailed' in kwargs.keys():
            SampleDetailed=kwargs['SampleDetailed']

        pTracking=0
        if 'pTracking' in kwargs.keys():
            pTracking=kwargs['pTracking']

        BackUp=0
        if 'BackUp' in kwargs.keys():
            BackUp=kwargs['BackUp']

        doSampling, doPlotting = [False, False]
        if 'plotPrms' in kwargs.keys():
            pStart, pStop, pStep = kwargs['plotPrms']
            doPlotting = True
            if init == True:
                tPlot = 0.0
                filenamelist = []
                timelist = []
            else:
                tPlot=G['iterFinalPlot']
                pStart = G['iterFinalPlot']+pStart+pStep
                pStop = G['iterFinalPlot']+pStop

        if 'samplePrms' in kwargs.keys():
            sStart, sStop, sStep = kwargs['samplePrms']
            doSampling = True
            if init == True:
                tSample = 0.0
                self._sampledict = {}
                filenamelistAvg = []
                timelistAvg = []
            else:
                tSample = G['iterFinalSample']
                sStart = G['iterFinalSample']+sStart+sStep
                sStop = G['iterFinalSample']+sStop


        t1 = ttime.time()
        if init:
            self._t = 0.0
            BackUpTStart=0.1*time
            BackUpT=0.1*time
            BackUpCounter=0
        else:
            self._t = G['dtFinal']
            self._tSample=G['iterFinalSample']
            time = G['dtFinal']+time
            BackUpT=0.1*time
            if 'BackUpCounter' in G.attributes():
                BackUpCounter=G['BackUpCounter']+1
                BackUpTStart=G['dtFinal']+BackUpT
            else:
                BackUpTStart=0.1*time
                BackUpCounter=0
            

        #Convert 'pBC' ['mmHG'] to default Units 
        G.vs['pBC']=[v*self._scaleToDef if v != None else None for v in G.vs['pBC']] 

        iteration=0
        t=self._t
        while True:
            if t >= time:
                break
            iteration += 1
            self._update_eff_resistance_and_LS(None, self._vi)
            self._solve(method, **kwargs)
            self._G.vs['pressure'] = deepcopy(self._x)
            self._update_flow_and_velocity()
            self._update_flow_sign()
            self._update_local_pressure_gradient()
            self._update_interface_vertices()
            if doPlotting and tPlot >= pStart and tPlot <= pStop:
                filename = 'iter_'+str(int(round(tPlot)))+'.vtp'
                #filename = 'iter_'+('%.3f' % t)+'.vtp'
                filenamelist.append(filename)
                timelist.append(tPlot)
                self._plot_rbc(filename)
                pStart = tPlot + pStep
            if pTracking and self._dt > 0:
                self._particleTracking()
            if SampleDetailed:
                self._sample()
                filenameDetailed ='G_iteration_'+str(iteration)+'.pkl'
                vgm.write_pkl(G,filenameDetailed)
            else:
                if doSampling and tSample >= sStart and tSample <= sStop:
                    self._t=t
                    sStart = tSample + sStep
                    self._sample()
                    if BackUp:
                        if t > BackUpTStart:
                            print('BackUp should be done')
                            print(BackUpCounter)
                            stdout.flush()
                            G['dtFinal']=t
                            G['iterFinalSample']=tSample
                            G['BackUpCounter']=BackUpCounter
                            filename1='sampledict_BackUp_'+str(BackUpCounter)+'.pkl'
                            filename2='G_BackUp'+str(BackUpCounter)+'.pkl'
                            #self._plot_sample_average()
                            #Convert deaultUnits to 'pBC' ['mmHG']
                            for v in G.vs:
                                if v['pBC'] != None:
                                    v['pBC']=v['pBC']/self._scaleToDef
                                v['pressure']=v['pressure']/self._scaleToDef
                            g_output.write_pkl(self._sampledict,filename1)
                            vgm.write_pkl(G,filename2)
                            #Convert 'pBC' ['mmHG'] to default Units
                            for v in G.vs:
                                if v['pBC'] != None:
                                    v['pBC']=v['pBC']*self._scaleToDef
                                v['pressure']=v['pressure']*self._scaleToDef
                            BackUpCounter += 1
                            BackUpTStart += BackUpT
            self._update_out_and_inflows_for_vertices()
            self._update_blocked_edges_and_timestep()

            self._propagate_rbc()
            tPlot = tPlot + self._dt
            self._tPlot = tPlot
            tSample = tSample + self._dt
            self._tSample = tSample
            t = t + self._dt
            log.info(t)
            stdout.write("\r%f" % tPlot)
            stdout.flush()
        stdout.write("\rDone. t=%f        \n" % tPlot)
        log.info("Time taken: %.2f" % (ttime.time()-t1))

        #Convert defaultUnits to 'pBC' ['mmHG']
        G.vs['pBC']=[v/self._scaleToDef if v != None else None for v in G.vs['pBC']]  

        if doPlotting:
            filename= 'iter_'+str(int(round(tPlot+1)))+'.vtp'
            filenamelist.append(filename)
            timelist.append(tPlot)
            self._plot_rbc(filename)
        G['dtFinal']=t
        G['iterFinalPlot']=tPlot
        G['iterFinalSample']=tSample
        if doSampling:
            self._sample()
            #self._plot_sample_average('sample_avg_final.vtp')
            g_output.write_pkl(self._sampledict, 'sampledict.pkl')
            if BackUp:
               filename1='sampledict_BackUp_'+str(BackUpCounter)+'.pkl'
               g_output.write_pkl(self._sampledict, filename1)
        G['dtFinal']=t
        G['iterFinalPlot']=tPlot
        G['iterFinalSample']=tSample
        vgm.write_pkl(G, 'G_final.pkl')
        vgm.write_pkl(self._RBCdict,'RBCdict.pkl')
        if BackUp:
            filename2='G_BackUp'+str(BackUpCounter)+'.pkl'
            vgm.write_pkl(G,filename2)

    #--------------------------------------------------------------------------

    def _plot_rbc(self, filename, tortuous=False):
        """Plots the current RBC distribution to vtp format.
        INPUT: filename: The name of the output file. This should have a .vtp
                         extension in order to be recognized by Paraview.
               tortuous: Whether or not to trace the tortuous path of the 
                         vessels. If false, linear tubes are assumed.
        OUTPUT: None, file written to disk.
        """
        G = self._G
        pgraph = vascularGraph.VascularGraph(0)
        r = []
        if tortuous:
            for e in G.es:
                if len(e['rRBC']) == 0:
                    continue
                p = e['points']
                cumlength = np.cumsum([np.linalg.norm(p[i] - p[i+1]) 
                                       for i in xrange(len(p[:-1]))])
                for rRBC in e['rRBC']:
                    i = np.nonzero(cumlength > rRBC)[0][0]
                    r.append(p[i-1] + (p[i] - p[i-1]) * 
                             (rRBC - cumlength[i-1]) / 
                             (cumlength[i] - cumlength[i-1]))
        else:
            for e in G.es:
                #points = e['points']
                #nPoints = len(points)
                rsource = G.vs[e.source]['r']
                dvec = G.vs[e.target]['r'] - G.vs[e.source]['r']
                length = e['length']
                for rRBC in e['rRBC']:
                    #index = int(round(npoints * rRBC / length))
                    r.append(rsource + dvec * rRBC/length)

	if len(r) > 0:
            pgraph.add_vertices(len(r))
            pgraph.vs['r'] = r
            g_output.write_vtp(pgraph, filename, False)
        else:
	    print('Network is empty - no plotting')

    #--------------------------------------------------------------------------
    
    def _sample(self):
        """Takes a snapshot of relevant current data and adds it to the sample
        database.
        INPUT: None
        OUTPUT: None, data added to self._sampledict
        """
        sampledict = self._sampledict
        G = self._G
        invivo = self._invivo
        
        htt2htd = self._P.tube_to_discharge_hematocrit
        du = self._G['defaultUnits']
       
        #Convert default units to ['mmHG'] 
        G.vs['pressure']=[v/self._scaleToDef for v in G.vs['pressure']]
 
        G.es['htd'] = [htt2htd(e['htt'], e['diameter'],invivo) for e in G.es]
        G.es['rbcFlow'] = [e['flow'] * e['htd'] for e in G.es]
        G.es['plasmaFlow'] = [e['flow'] - e['rbcFlow'] for e in G.es]
        G.es['nRBC'] = [len(e['rRBC']) for e in G.es]
	G.es['lpg']=np.array(G.es['specificResistance']) * np.array(G.es['flow'])
        
        for eprop in ['flow', 'v', 'htt', 'htd', 'effResistance', 
                      'rbcFlow', 'plasmaFlow', 'nRBC','lpg']:
            if not eprop in sampledict.keys():
                sampledict[eprop] = []
            sampledict[eprop].append(G.es[eprop])
        for vprop in ['pressure']:
            if not vprop in sampledict.keys():
                sampledict[vprop] = []
            sampledict[vprop].append(G.vs[vprop])
        if not 'time' in sampledict.keys():
            sampledict['time'] = []
        sampledict['time'].append(self._tSample)

        #Convert ['mmHG'] to default units
        G.vs['pressure']=[v*self._scaleToDef for v in G.vs['pressure']] 

    #--------------------------------------------------------------------------

    def _particleTracking(self):
        """Takes a snapshot of relevant current data and adds it to the sample
        database.
        INPUT: None
        OUTPUT: None, data added to self._sampledict
        """
        RBCdict = self._RBCdict
        G = self._G
        newRBCs = self._newRBCs
        tSample = self._tSample

        initRBCdict = self._initRBCdict
        if initRBCdict:
            for eprop in ['flow','v','sign']:
                if eprop not in RBCdict.keys():
                    RBCdict[eprop]=[]
                    for i in range(G.ecount()):
                        RBCdict[eprop].append([])
            if 'time' not in RBCdict.keys():
                RBCdict['time']=[]
            self._initRBCdict = 0
         
        if newRBCs != []:
            for i in newRBCs:
                RBCdict[i]=[]
                RBCdict[i].append([]) #time
                RBCdict[i].append([]) #edge
                RBCdict[i].append([]) #position
            newRBCs = []

        for e in G.es:
            eI = e.index
            sign = e['sign']
            if sign == 1:
                diff = 0
            else: 
                diff = len(e['rRBC'])-len(e['RBCindex'])
            if len(e['RBCindex']) > 0:
                for k,j in enumerate(e['RBCindex']):
                    index = k + diff
                    RBCdict[j][0].append(tSample)
                    RBCdict[j][1].append(eI)
                    RBCdict[j][2].append(e['rRBC'][index])

        for eprop in ['flow','v','sign']:
            for e in G.es:
                RBCdict[eprop][e.index].append(e[eprop])
        RBCdict['time'].append(tSample)
                
    #--------------------------------------------------------------------------

    def _plot_sample_average(self, sampleAvgFilename):
        """Averages the self._sampleDict data and writes it to disc.
        INPUT: sampleAvgFilename: Name of the sample average out-file.
        OUTPUT: None
        """
        sampledict = self._sampledict
        G = self._G

        #Convert default units to ['mmHG'] 
        G.vs['pressure']=[v/self._scaleToDef for v in G.vs['pressure']]
       
        for eprop in ['flow', 'v', 'htt', 'htd', 
                      'rbcFlow', 'plasmaFlow', 'nRBC']:
            G.es[eprop + '_avg'] = np.average(sampledict[eprop], axis=0)
            if not [eprop + '_avg'] in sampledict.keys():
                sampledict[eprop + '_avg']=[]
            sampledict[eprop + '_avg'].append(G.es[eprop + '_avg'])
        for vprop in ['pressure']:
            G.vs[vprop + '_avg'] = np.average(sampledict[vprop], axis=0)
            if not [vprop + '_avg'] in sampledict.keys():
                sampledict[vprop + '_avg']=[]
            sampledict[vprop + '_avg'].append(G.vs[vprop + '_avg'])
        g_output.write_vtp(G, sampleAvgFilename, False)

        #Convert ['mmHG'] to default units
        G.vs['pressure']=[v*self._scaleToDef for v in G.vs['pressure']]

    #--------------------------------------------------------------------------

    def _solve(self, method, **kwargs):
        """Solves the linear system A x = b using a direct or AMG solver.
        INPUT: method: This can be either 'direct' or 'iterative'
               **kwargs
               precision: The accuracy to which the ls is to be solved. If not
                          supplied, machine accuracy will be used. (This only
                          applies to the iterative solver)
        OUTPUT: None, self._x is updated.
        """
        A = self._A.tocsr()
        if method == 'direct':
            linalg.use_solver(useUmfpack=True)
            x = linalg.spsolve(A, self._b)
        elif method == 'iterative':
            if kwargs.has_key('precision'):
                eps = kwargs['precision']
            else:
                eps = finfo(float).eps
            #AA = ruge_stuben_solver(A)
            AA = smoothed_aggregation_solver(A, max_levels=10, max_coarse=500)
            #PC = AA.aspreconditioner(cycle='V')
            #x,info = linalg.cg(A, self._b, tol=eps, maxiter=30, M=PC)
            #(x,flag) = pyamg.krylov.fgmres(A,self._b, maxiter=30, tol=eps)
            x = abs(AA.solve(self._b, tol=eps, accel='cg')) # abs required, as (small) negative pressures may arise
        self._x = x

    #--------------------------------------------------------------------------

    def _verify_mass_balance(self):
        """Computes the mass balance, i.e. sum of flows at each node and adds
        the result as a vertex property 'flowSum'.
        INPUT: None
        OUTPUT: None (result added as vertex property)
        """
        G = self._G
        G.vs['flowSum'] = [sum([G.es[e]['flow'] * np.sign(G.vs[v]['pressure'] -
                                                    G.vs[n]['pressure'])
                               for e, n in zip(G.adjacent(v), G.neighbors(v))])
                           for v in xrange(G.vcount())]

    #--------------------------------------------------------------------------

    def _verify_rbc_balance(self):
        """Computes the rbc balance, i.e. sum of rbc flows at each node and
        adds the result as a vertex property 'rbcFlowSum'.
        INPUT: None
        OUTPUT: None (result added as vertex property)
        """
        G = self._G
        vf = self._P.velocity_factor
        invivo=self._invivo
        lrbc = self._P.effective_rbc_length
        tubeHt = [0.0 if e['tubeHt'] is None else e['tubeHt'] for e in G.es]
        G.vs['rbcFlowSum'] = [sum([4.0 * G.es[e]['flow'] * vf(G.es[e]['diameter'],invivo) * tubeHt[e] /
                                   np.pi / G.es[e]['diameter']**2 / lrbc(G.es[e]['diameter']) *
                                   np.sign(G.vs[v]['pressure'] - G.vs[n]['pressure'])
                                   for e, n in zip(G.adjacent(v), G.neighbors(v))])
                              for v in xrange(G.vcount())]

    #--------------------------------------------------------------------------

    def _verify_p_consistency(self):
        """Checks for local pressure maxima at non-pBC vertices.
        INPUT: None.
        OUTPUT: A list of local pressure maxima vertices and the maximum 
                pressure difference to their respective neighbors."""
        G = self._G
        localMaxima = []
        #Convert 'pBC' ['mmHG'] to default Units
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']*self._scaleToDef

        for i, v in enumerate(G.vs):
            if v['pBC'] is None:
                pdiff = [v['pressure'] - n['pressure']
                         for n in G.vs[G.neighbors(i)]]
                if min(pdiff) > 0:
                    localMaxima.append((i, max(pdiff)))         
        #Convert defaultUnits to 'pBC' ['mmHG']
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']/self._scaleToDef

        return localMaxima

    #--------------------------------------------------------------------------
    
    def _residual_norm(self):
        """Computes the norm of the current residual.
        """
        return np.linalg.norm(self._A * self._x - self._b)
