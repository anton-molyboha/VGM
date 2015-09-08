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

__all__ = ['LinearSystemConti']
log = vgm.LogDispatcher.create_logger(__name__)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class LinearSystemConti(object):
    """Implements and extends the discrete red blood cell transport as proposed
    by Obrist and coworkers (2010).
    """
    def __init__(self, G, invivo=True, dThreshold=10.0,  assert_well_posedness=True,
                 init=True,**kwargs):
        """Initializes a LinearSystemHtd instance.
        INPUT: G: Vascular graph in iGraph format.
               invivo: Boolean, whether the physiological blood characteristics 
                       are calculated using the invivo (=True) or invitro (=False)
                       equations
               dThreshold: Diameter threshold below which vessels are
                           considered capillaries, to which the reordering
                           algorithm can be applied.
               assert_well_posedness: (Optional, default=True.) Boolean whether
                                      or not to check components for correct
                                      pressure boundary conditions.
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
        self._tSample = 0.0
        self._t=0.0
        self._BackUpCounter=0
        self._filenamelist = []
        self._timelist = []
        self._filenamelistAvg = []
	self._timelistAvg = []
        self._sampledict = {} 
	self._transitTimeDict = {}
	self._init=init
	self._convergence = False
	htd2htt=self._P.discharge_to_tube_hematocrit
        htt2htd = self._P.tube_to_discharge_hematocrit
	self._dPhase=1.0
	print('self._dPhase')
	print(self._dPhase)
	
        # Assure that both pBC and rBC edge properties are present:
        for key in ['pBC', 'rBC']:
            if not G.vs[0].attributes().has_key(key):
                G.vs[0][key] = None

        # Set initial pressure and flow to zero:
        if init:	
            G.vs['pressure'] = [0.0 for v in G.vs]
            G.es['flow'] = [0.0 for e in G.es]
	    G.es['flowRBCIn']=[None for e in G.es]

	#Read sampledict (must be in folder, from where simulation is started)
	if not init:
	   self._sampledict=vgm.read_pkl('sampledict.pkl')
	   self._t=G['dtFinal']
           self._tSample= G['iterFinalSample']
           self._BackUpCounter=G['BackUpCounter']
	
	#Calculate total network Volume
        G['V']=0
	for e in G.es:
	    G['V']=G['V']+0.25*np.pi*e['diameter']**2*e['length']

        # Set initial bifurcation vertex, in-edge and timestep:
        self._vi = None
        self._eiIn = None
        self._dt = -1
        self._tlist = dict([(i, 1e25) for i in xrange(G.ecount())])
        
        # Compute the edge-specific minimal RBC distance:
        vrbc = self._P.rbc_volume()
	
	#Calculate Length of RBCs, nMax, httMax (which come from the pries equation to change tube to discharge hematocrit)
        G.es['minDist'] = [vrbc /(np.pi * e['diameter']**2 / 4) for e in G.es]
        G.es['nMax'] =[e['length'] / e['minDist'] for e in G.es]
	G.es['httMax']=[htd2htt(0.99, e['diameter'], True) for e in G.es]

        #G.es['nMax'] = [np.pi * e['diameter']**2 / 4 * e['length'] / vrbc 
        #                for e in G.es]
        #G.es['minDist'] = [e['length'] / e['nMax'] for e in G.es]

        # Group edges in capillary and noncapillary:
#        self._capEdgesI = G.es(diameter_le=dThreshold).indices
#        self._noncapEdgesI = np.setdiff1d(xrange(G.ecount()), self._capEdgesI).tolist()

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
                    Nmax = max(e['nMax'], 1)
                    if e['httBC'] is not None:
                        N = e['httBC'] * Nmax
                    else:
                        if kwargs.has_key('hd0'):
            		    ht0=self._P.discharge_to_tube_hematocrit(hd0,e['diameter'],invivo)
                        N = ht0 * Nmax
                    e['nRBC']=N                    
        if kwargs.has_key('plasmaViscosity'):
            self._muPlasma = kwargs['plasmaViscosity']
        else:
            self._muPlasma = self._P.dynamic_plasma_viscosity()

        # Compute nominal and specific resistance:
        self._update_nominal_and_specific_resistance()

        # Compute the current tube hematocrit from the RBC positions:
	for e in G.es:
            e['htt']=e['nRBC']*vrbc/(0.25*np.pi*e['diameter']**2*e['length'])
            e['htd']=min(htt2htd(e['htt'], e['diameter'], invivo), 0.99)
 
        # This initializes the full LS. Later, only relevant parts of
        # the LS need to be changed at any timestep. Also, RBCs are
        # removed from no-flow edges to avoid wasting computational
        # time on non-functional vascular branches / fragments:

        self._update_eff_resistance_and_LS(None, None, assert_well_posedness)
        self._solve('direct')
        self._G.vs['pressure'] = deepcopy(self._x)
        self._update_flow_and_velocity()
        G.es['rbcFlow']=[e['htd']*e['flow'] for e in G.es()]
        G.es['flowRBCTube']=[e['rbcFlow']/vrbc for e in G.es()]
        #self._verify_mass_balance()
        #self._verify_rbc_balance()
	self._update_rbcFlowRates()
	self._update_hematocrit()
	#print('init')
	#print(G.es['htt'])
        for v in G.vs:
            v['pressure']=v['pressure']/vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])

	#Calculate an estimated network turnover time (based on conditions at the beginning)
        flowsum=0
	for vi in G['av']:
	    for ei in G.adjacent(vi):
               flowsum=flowsum+G.es['flow'][ei]
        G['Ttau']=G['V']/flowsum
        stdout.write("\rEstimated network turnover time Ttau=%f        \n" % G['Ttau'])

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

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)
        G.es['specificResistance'] = [128 * self._muPlasma / (np.pi * d**4)
                                        for d in G.es['diameter']]

        G.es['resistance'] = [l * sr for l, sr in zip(G.es['length'], 
                                                G.es['specificResistance'])]

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
        G.es['sign'] = [np.sign(G.vs[e.source]['pressure'] -
                                G.vs[e.target]['pressure']) for e in G.es]

    #--------------------------------------------------------------------------

    def _update_flow_and_velocity(self):
        """Updates the flow and red blood cell velocity in all vessels
        INPUT: None
        OUTPUT: None
        """
        G = self._G
        invivo=self._invivo
        vf = self._P.velocity_factor
        vrbc = self._P.rbc_volume()
        G.es['flow'] = [abs(G.vs[e.source]['pressure'] -
                            G.vs[e.target]['pressure']) / e['effResistance']
                        for e in G.es]
        G.es['rbcFlow'] =[e['flow'] * e['htd'] for e in G.es]
        G.es['flowRBCTube']=[e['rbcFlow']/vrbc for e in G.es]
        # RBC velocity is not defined if tube_ht==0, using plasma velocity
        # instead:
        G.es['v'] = [4 * e['flow'] * vf(e['diameter'], invivo, tube_ht=e['htt']) /
                     (np.pi * e['diameter']**2) if e['htt'] > 0 else
                     4 * e['flow'] / (np.pi * e['diameter']**2)
                     for e in G.es]


    #--------------------------------------------------------------------------

    def _update_eff_resistance_and_LS(self, newGraph=None, vertex=None,
                                      assert_well_posedness=True):
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
               assert_well_posedness: (Optional, default=True.) Boolean whether
                                      or not to check components for correct
                                      pressure boundary conditions.
        OUTPUT: A: Matrix A of the linear system, holding the conductance
                   information.
                b: Vector b of the linear system, holding the boundary
                   conditions.
        """

        if newGraph is not None:
            self._G = newGraph

        G = self._G
        P = self._P
        A = self._A
        b = self._b
        x = self._x
        invivo = self._invivo

        htt2htd = P.tube_to_discharge_hematocrit
        nurelCompare = P.relative_apparent_blood_viscosity
        nurel = P.relative_apparent_blood_viscosityTest

        if assert_well_posedness:
            # Ensure that the problem is well posed in terms of BCs.
            # This takes care of unconnected nodes as well as connected
            # components of the graph that have not been assigned a minimum of
            # one pressure boundary condition:
            for component in G.components():
                if all([x is None for x in G.vs(component)['pBC']]):
                    i = component[0]
                    G.vs[i]['pBC'] = 0.0

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


        for e in edgeList:
            d = e['diameter']
            # Compute discharge hematocrit.
            # Ensure htd < 1 to avoid excessively high resistance:
            dischargeHt = min(htt2htd(e['htt'], d, invivo), 1.00)
	    if dischargeHt == 1.0:
	        print('ATTENTION_HTD of 1, reached in Vessel')
	        print(e.index)
            nuCompare = nurelCompare(d, 0.99, invivo)
            nu = nurel(d, dischargeHt, 1*nuCompare,invivo)
	    #print('edge')
	    #print(e['htt'])
	    #print(nu)
	    #print(dischargeHt)
            e['effResistance'] = e['resistance'] * nu

            # TODO: come up with a more clever strategy of increasing
            # the resistance of blocked vessels.
            #if e.index in self._blockedEdges:
            #    e['effResistance'] = e['effResistance'] * 100

        #Convert 'pBC' ['mmHG'] to default Units
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']*vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])
	#print('LS')
	#print(G.es['effResistance'])

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
                    #elif neighbor['rBC'] is not None:
                     #   b[i] = b[i] + neighbor['rBC']
                    else:
                        if j not in neighbors:
                            A[i,j] = - conductance
                        else:
                            A[i,j] = A[i,j] - conductance
                    neighbors.append(j)
                    if vertex['rBC'] is not None:
                        b[i] += vertex['rBC']
                A[i,i]=aDummy

        #Convert deaultUnits to 'pBC' ['mmHG']
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']/vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])


        self._A = A
        self._b = b
	self._G = G

    #--------------------------------------------------------------------------

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
	       init: boolean whether evolve is a restart or not.
	       dt: timestep
	       dPhase: Diameter to define the Phase separation curve
         OUTPUT: None (files are written to disk)
        """
        G=self._G
        tSample = self._tSample
        filenamelist = self._filenamelist
        timelist = self._timelist
	filenamelistAvg = self._filenamelistAvg
	timelistAvg = self._timelistAvg
	
	#print('Evolve')
	#print(G.vs['pressure'])

	if 'dt' in kwargs.keys():
	    self._dt=kwargs['dt']
	else:
	    self._dt=0.01

	if 'dPhase' in kwargs.keys():
	    self._dPhase=kwargs['dPhase']
	else:
	    self._dPhase=1.0

	if 'init' in kwargs.keys():
	    init=kwargs['init']
	else:
	    init=self._init

        SampleDetailed=False
        if 'SampleDetailed' in kwargs.keys():
            SampleDetailed=kwargs['SampleDetailed']

        doSampling = [False]

        if 'samplePrms' in kwargs.keys():
            sStart, sStop, sStep = kwargs['samplePrms']
            doSampling = True
            if init == True:
                tSample = 0.0
                self._sampledict = {}
                self._sampledict['nRBCConvergence']=[]
                self._sampledict['flowConvergence']=[]
                self._sampledict['RBCFlowConvergence']=[]
		self._transitTimeDict = {}
                filenamelistAvg = []
                timelistAvg = []
            else:
		tSample = G['iterFinalSample']
                sStart = G['iterFinalSample']+sStart+sStep
                sStop = G['iterFinalSample']+sStop

        t1 = ttime.time()
        if init:
            t = 0.0
            tBackUp=0.05*time
            BackUpCounter=0
        else:
            t = G['dtFinal']
            time = G['dtFinal']+time
            BackUpCounter=G['BackUpCounter']
            tBackUp=0.05*t+BackUpCounter*0.05*t

        iteration=0
        turnoverCounter=0
	flowCounter=0
        while True:
	    print('NEW_ITERATION')
            if t >= time:
                break
            iteration += 1
            self._update_eff_resistance_and_LS(None, self._vi, False)
            self._solve(method, **kwargs)
            self._G.vs['pressure'] = deepcopy(self._x)
            self._update_flow_and_velocity()
            self._update_flow_sign()
            for v in G.vs:
                v['pressure']=v['pressure']/vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])
            self._update_interface_vertices()
            if SampleDetailed:
                self._sample()
                filenameDetailed ='G_iteration_'+str(iteration)+'.pkl'
                vgm.write_pkl(G,filenameDetailed)
            else:
                if doSampling and tSample >= sStart and tSample <= sStop:
                    self._sample()
		    self._t=t
		    self._tSample=tSample
		    self._BackUpCounter=BackUpCounter
                    sStart = tSample + sStep
		    flowCounter += 1
		    if flowCounter == 50:
                        self._verify_mass_balance()
                        self._verify_rbc_balance()
			flowCounter =0
		    self._convergence_check()
	    if self._convergence == True:
		break
            if doSampling and t >= (turnoverCounter + 1)*G['Ttau']:
		filenameAvg = 'sample_avg_'+str(int(turnoverCounter))+'.vtp'
		filenamelistAvg.append(filenameAvg)
		timelistAvg.append(tSample)
		self._plot_sample_average(filenameAvg)
		turnoverCounter += 1
	    if t > tBackUp:
		print('BackUp should be done')
		print(BackUpCounter)
		G['dtFinal']=t
                G['iterFinalSample']=tSample
                G['BackUpCounter']=BackUpCounter
		filename1='sampledict_BackUp_'+str(BackUpCounter)+'.pkl'
                filename2='G_BackUp'+str(BackUpCounter)+'.pkl'
		print(filename1)
		print(filename2)
		g_output.write_pkl(self._sampledict,filename1)
		vgm.write_pkl(G,filename2)
		BackUpCounter += 1
		tBackUp += 0.05*time

	    self._update_rbcFlowRates()
	    self._update_hematocrit()	
            tSample = tSample + self._dt
            self._tSample = tSample
            t = t + self._dt
	    log.info(t)
            stdout.write("\rt=%f  \n" % tSample)
            stdout.flush()
        stdout.write("\rDone. t=%f        \n" % tSample)
        log.info("Time taken: %.2f" % (ttime.time()-t1))

        self._t=t
        self._BackUpCounter=BackUpCounter
        self._update_flow_and_velocity
        self._verify_mass_balance()
        self._verify_rbc_balance()
        
	stdout.flush()

        if doSampling:
            self._sample()
            self._plot_sample_average('sample_avg_final.vtp')
            g_output.write_pkl(self._sampledict, 'sampledict.pkl')
	    g_output.write_pkl(self._transitTimeDict, 'TransitTimeDict.pkl')
            g_output.write_pvd_time_series('sequenceSampling.pvd',
					   filenamelistAvg, timelistAvg)
	G['dtFinal']=t
	G['iterFinalSample']=tSample
        G['BackUpCounter']=BackUpCounter
        vgm.write_pkl(G, 'G_final.pkl')
        # Since Physiology has been rewritten using Cython, it cannot be
        # pickled. This class holds a Physiology object as a member and
        # consequently connot be pickled either.
        #g_output.write_pkl(self, 'LSHTD.pkl')
        self._timelist = timelist[:]
        self._filenamelist = filenamelist[:]
	self._filenamelistAvg = filenamelistAvg[:]
	self._timelistAvg = timelistAvg[:]

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
        
        #G.es['htd'] = [htt2htd(e['htt'], e['diameter'],invivo) for e in G.es]
        #G.es['rbcFlow'] = [e['flow'] * e['htd'] for e in G.es]
        G.es['plasmaFlow'] = [e['flow'] - e['rbcFlow'] for e in G.es]
        
        for eprop in ['flow', 'v', 'htt', 'htd', 'effResistance', 
                      'rbcFlow', 'plasmaFlow', 'nRBC','flowRBCTube','flowRBCIn','sign']:
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

	self._sampledict = sampledict

    #--------------------------------------------------------------------------

    def _plot_sample_average(self, sampleAvgFilename):
        """Averages the self._sampleDict data and writes it to disc.
        INPUT: sampleAvgFilename: Name of the sample average out-file.
        OUTPUT: None
        """
        sampledict = self._sampledict
        G = self._G
        
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
	#print('Solve')
	#print(self._x)

    #--------------------------------------------------------------------------

    def _verify_mass_balance(self):
        """Computes the mass balance, i.e. sum of flows at each node and adds
        the result as a vertex property 'flowSum'.
        INPUT: None
        OUTPUT: None (result added as vertex property)
        """
        G = self._G
	t=self._t
	tSample=self._tSample
        BackUpCounter=self._BackUpCounter
	inAndOut=[110,131]
	sampledict=self._sampledict
        for i in range(G.vcount()):
            G.vs[i]['flowSum'] = sum([G.es[e]['flow'] * np.sign(G.vs[i]['pressure'] -
                                                    G.vs[n]['pressure'])
                               for e, n in zip(G.adjacent(i), G.neighbors(i))])
	    if abs(np.round(G.vs[i]['flowSum']*1e11)/1e11) > 0 and i not in inAndOut:
                print('ERROR flowSum not equal to 0')
                print(i)
                print(G.vs[i]['flowSum'])
                G['dtFinal']=t
                G['iterFinalSample']=tSample
                G['BackUpCounter']=BackUpCounter
		vgm.write_pkl(sampledict,'sampledict_massFlowERROR.pkl')
		vgm.write_pkl(G,'G_massFlowERROR.pkl')

    #--------------------------------------------------------------------------

    def _verify_rbc_balance(self):
        """Computes the rbc balance, i.e. sum of rbc flows at each node and
        adds the result as a vertex property 'rbcFlowSum'.
        INPUT: None
        OUTPUT: None (result added as vertex property)
        """
        G = self._G
        t=self._t
        tSample=self._tSample
        BackUpCounter=self._BackUpCounter
	vrbc=self._P.rbc_volume()
        inAndOut=[110,131]
        sampledict=self._sampledict
	for i in range(G.vcount()):
            G.vs[i]['rbcFlowSum'] = sum([G.es[e]['rbcFlow'] * np.sign(G.vs[i]['pressure'] -
                                                    G.vs[n]['pressure'])
                               for e, n in zip(G.adjacent(i), G.neighbors(i))])
            if abs(np.round(G.vs[i]['rbcFlowSum']*1e11)/1e11) > 0 and i not in inAndOut:
                print('ERROR rbcFlowSum not equal to 0')
	        print(i)
	        print(G.vs[i]['rbcFlowSum'])
                G['dtFinal']=t
                G['iterFinalSample']=tSample
                G['BackUpCounter']=BackUpCounter
                vgm.write_pkl(sampledict,'sampledict_rbcFlowERROR.pkl')
                vgm.write_pkl(G,'G_rbcFlowERROR.pkl')

    #-------------------------------------------------------------------------

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
                v['pBC']=v['pBC']*vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])

        for i, v in enumerate(G.vs):
            if v['pBC'] is None:
                pdiff = [v['pressure'] - n['pressure']
                         for n in G.vs[G.neighbors(i)]]
                if min(pdiff) > 0:
                    localMaxima.append((i, max(pdiff)))         
        #Convert defaultUnits to 'pBC' ['mmHG']
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']/vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])

        return localMaxima

    #--------------------------------------------------------------------------
    
    def _residual_norm(self):
        """Computes the norm of the current residual.
        """
        return np.linalg.norm(self._A * self._x - self._b)

    #--------------------------------------------------------------------------

    def _update_rbcFlowRates(self):
        """Computes the rbcFlowRates for every edge, based on the "S-curve" 
	   only valid for 3-vessels at a bifurcation!
        """

	G = self._G
	eps=self._eps
        dt = self._dt
        pse = self._P.phase_separation_effect_steep
        vrbc = self._P.rbc_volume()
	dPhase=self._dPhase
	G.es['flowRBCMax']=None
	VertexStart=None
	
        pSortedVertices = sorted([(v['pressure'], v.index) for v in G.vs],
                                 reverse=True)

	#print(pSortedVertices)

	pSortedVerticesLoop = pSortedVertices

        for vertexPressure, vertex in pSortedVerticesLoop:
           outEdges = []
           inEdges = []
	   #print(vertex)
	   #Define in- and outEdges
           for neighbor, edge in zip(G.neighbors(vertex, 'all'), G.adjacent(vertex, 'all')):
                if G.vs[neighbor]['pressure'] < vertexPressure - eps:
                    outEdges.append(edge)
                elif G.vs[neighbor]['pressure'] > vertexPressure + eps:
                    inEdges.append(edge)
		else :
		   print('Equal Pressure: There is no flow in edge')
		   print(edge)
		   G.es[edge]['flowRBCTube']=0
           print('inEdges')
           print(inEdges)
           print('outEdges')
           print(outEdges)

           #Distribute RBC Flow    
           if len(outEdges) > 0:
               #Inlet
               if len(inEdges) == 0 and len(outEdges) ==1:
                   G.es[outEdges[0]]['flowRBCTube']=G.es[outEdges[0]]['flowRBCTube']
		   G.es[outEdges[0]]['flowRBCIn']=G.es[outEdges[0]]['flowRBCTube']
                   print(G.es[outEdges[0]]['flowRBCTube'])
               #It s a convergent bifurcation
               elif len(inEdges) > len(outEdges):
                   G.es[outEdges[0]]['flowRBCTube']=(G.es[inEdges[0]]['flowRBCTube']+G.es[inEdges[1]]['flowRBCTube'])
                   print(G.es[outEdges[0]]['flowRBCTube'])
                   print(G.es[inEdges[0]]['flowRBCTube'])
                   print(G.es[inEdges[1]]['flowRBCTube'])
                   G.es[outEdges[0]]['flowRBCIn']=G.es[outEdges[0]]['flowRBCTube']
               #There is exactly one in- and one outEdge
               elif len(inEdges) == len(outEdges):
                   G.es[outEdges[0]]['flowRBCTube']=G.es[inEdges[0]]['flowRBCTube']
                   G.es[outEdges[0]]['flowRBCIn']=G.es[outEdges[0]]['flowRBCTube']
                   print(G.es[outEdges[0]]['flowRBCTube'])
               #It s divergent bifurcation
               else:
		   if G.es[inEdges[0]]['flowRBCTube']*vrbc > G.es[outEdges[0]]['rbcFlow']+G.es[outEdges[1]]['rbcFlow']:
		       print('RBC_FLOW Increase!')
		       print(G.es[inEdges[0]]['flowRBCTube']*vrbc)
                       print(G.es[outEdges[0]]['rbcFlow']+G.es[outEdges[1]]['rbcFlow'])
                   print(G.es[outEdges[0]]['flow']/G.es[inEdges[0]]['flow'])
                   rbcFlowRel=pse(G.es[outEdges[0]]['flow']/G.es[inEdges[0]]['flow'],dPhase)
                   print(rbcFlowRel)
                   G.es[outEdges[0]]['flowRBCTube']=rbcFlowRel*G.es[inEdges[0]]['flowRBCTube']
                   G.es[outEdges[1]]['flowRBCTube']=(1-rbcFlowRel)*G.es[inEdges[0]]['flowRBCTube']
                   print(G.es[inEdges[0]]['flowRBCTube'])
                   print(G.es[outEdges[0]]['flowRBCTube'])
                   print(G.es[outEdges[1]]['flowRBCTube'])
                   G.es[outEdges[0]]['flowRBCIn']=G.es[outEdges[0]]['flowRBCTube']
                   G.es[outEdges[1]]['flowRBCIn']=G.es[outEdges[1]]['flowRBCTube']
		   if G.es[outEdges[0]]['htd']==1 or G.es[outEdges[1]]['htd']==1:
		       print('CHECK_IT')
		       print(G.es[outEdges[0]]['htd'])
                       print(G.es[outEdges[1]]['htd'])
  		       print('htd mother vessel')
		       print(G.es[inEdges[0]]['htd'])
		       if G.es[outEdges[0]]['htd']==1 and G.es[outEdges[1]]['htd']==1:
		           print('2 Constraints')
                           print('Flow in that Edges is equal to =')
                           print(G.es[outEdges[0]]['flow'])
                           print(G.es[outEdges[1]]['flow'])
                           print('RBC Flow in Mother Vessel is equal to')
                           print(G.es[inEdges[0]]['flowRBCTube']*vrbc)
                           print('Phase Separation Factor is equal to')
                           print(rbcFlowRel)
                           print('The resulting RBC Flow in the daughter vessels is equal to')
                           print(G.es[outEdges[0]]['flowRBCIn']*vrbc)
                           print(G.es[outEdges[1]]['flowRBCIn']*vrbc)
		           if G.es[outEdges[0]]['flowRBCIn']*vrbc > G.es[outEdges[0]]['flow']:
                               print(G.es[outEdges[0]]['flow'])
                               print(G.es[outEdges[1]]['flow'])
                               print(G.es[outEdges[0]]['flow']/vrbc)
                               print(G.es[outEdges[1]]['flow']/vrbc)
			       G.es[outEdges[0]]['flowRBCIn']=G.es[outEdges[0]]['flow']/vrbc
			       G.es[outEdges[1]]['flowRBCIn']=G.es[inEdges[0]]['flow']/vrbc-G.es[outEdges[0]]['flowRBCIn']
			       G.es[outEdges[0]]['flowRBCTube']=G.es[outEdges[0]]['flowRBCIn']
                               G.es[outEdges[1]]['flowRBCTube']=G.es[outEdges[1]]['flowRBCIn']
                               print('LIMIT: RBC Flow')
                               print(G.es[outEdges[0]]['flowRBCIn']*vrbc)
                               print(G.es[outEdges[1]]['flowRBCIn']*vrbc)  
			       if np.floor(G.es[outEdges[1]]['flowRBCIn']*10**14)/10**14 > np.floor(G.es[outEdges[1]]['flow']/vrbc*10**14)/10**14:
			           print('MEGA_WARNING')
				   print(G.es[outEdges[1]]['flowRBCIn'])
				   print(G.es[outEdges[1]]['flow']/vrbc)
			   elif G.es[outEdges[1]]['flowRBCIn']*vrbc > G.es[outEdges[1]]['flow']:
                               print(G.es[outEdges[0]]['flow'])
                               print(G.es[outEdges[1]]['flow'])
                               print(G.es[outEdges[0]]['flow']/vrbc)
                               print(G.es[outEdges[1]]['flow']/vrbc)
                               G.es[outEdges[1]]['flowRBCIn']=G.es[outEdges[1]]['flow']/vrbc
                               G.es[outEdges[0]]['flowRBCIn']=G.es[inEdges[0]]['flow']/vrbc-G.es[outEdges[1]]['flowRBCIn']
                               G.es[outEdges[0]]['flowRBCTube']=G.es[outEdges[0]]['flowRBCIn']
                               G.es[outEdges[1]]['flowRBCTube']=G.es[outEdges[1]]['flowRBCIn']
                               print('LIMIT: RBC Flow')
                               print(G.es[outEdges[1]]['flowRBCIn']*vrbc)
                               print(G.es[outEdges[0]]['flowRBCIn']*vrbc)    
                               if np.floor(G.es[outEdges[0]]['flowRBCIn']*10**14)/10**14 > np.floor(10**14*G.es[outEdges[1]]['flow']/vrbc)/10**14:
                                   print('MEGA_WARNING')
                                   print(G.es[outEdges[0]]['flowRBCIn'])
                                   print(G.es[outEdges[0]]['flow']/vrbc)
		       elif G.es[outEdges[0]]['htd']==1 and G.es[outEdges[1]]['htd'] < 1:
			   print('htd is equal to 1 in Edge')
			   print(outEdges[0])
			   print('Flow in that Edge is equal to =')
			   print(G.es[outEdges[0]]['flow'])
			   print('RBC Flow in Mother Vessel is equal to')
                           print(G.es[inEdges[0]]['flowRBCTube']*vrbc)
			   print('Phase Separation Factor is equal to')
                           print(rbcFlowRel)
			   print('The resulting RBC Flow in the daughter vessel is equal to')
			   print(G.es[outEdges[0]]['flowRBCIn']*vrbc)
			   if G.es[outEdges[0]]['flowRBCIn']*vrbc > G.es[outEdges[0]]['flow']:
                               G.es[outEdges[0]]['flowRBCTube']=G.es[outEdges[0]]['flow']/vrbc
                               G.es[outEdges[1]]['flowRBCTube']=G.es[inEdges[0]]['flowRBCTube']-G.es[outEdges[0]]['flowRBCTube']
                               G.es[outEdges[0]]['flowRBCIn']=G.es[outEdges[0]]['flowRBCTube']
                               G.es[outEdges[1]]['flowRBCIn']=G.es[outEdges[1]]['flowRBCTube']
			       print('LIMIT: RBC Flow')
			       print(G.es[outEdges[0]]['flowRBCTube']*vrbc)
                               print(G.es[outEdges[1]]['flowRBCTube']*vrbc)     
                       elif G.es[outEdges[1]]['htd']==1 and G.es[outEdges[0]]['htd'] < 1:
                           print('htd is equal to 1 in Edge')
                           print(outEdges[1])
                           print('Flow in that Edge is equal to =')
                           print(G.es[outEdges[1]]['flow'])
                           print('RBC Flow in Mother Vessel is equal to')
                           print(G.es[inEdges[0]]['flowRBCTube']*vrbc)
                           print('Phase Separation Factor is equal to')
                           print(rbcFlowRel)
                           print('The resulting RBC Flow in the daughter vessel is equal to')
                           print(G.es[outEdges[1]]['flowRBCIn']*vrbc)
                           if G.es[outEdges[1]]['flowRBCIn']*vrbc > G.es[outEdges[1]]['flow']:
                               G.es[outEdges[1]]['flowRBCTube']=G.es[outEdges[1]]['flow']/vrbc
                               G.es[outEdges[0]]['flowRBCTube']=G.es[inEdges[0]]['flowRBCTube']-G.es[outEdges[1]]['flowRBCTube']
                               G.es[outEdges[0]]['flowRBCIn']=G.es[outEdges[0]]['flowRBCTube']
                               G.es[outEdges[1]]['flowRBCIn']=G.es[outEdges[1]]['flowRBCTube']
                               print('LIMIT: RBC Flow')
                               print(G.es[outEdges[1]]['flowRBCTube']*vrbc)     
                               print(G.es[outEdges[0]]['flowRBCTube']*vrbc)

#		#Distribute RBC Flow	
#	        if len(outEdges) > 0:
#		    #Inlet
#		    if len(inEdges) == 0 and len(outEdges) ==1:
#    		        G.es[outEdges[0]]['flowRBCTube']=G.es[outEdges[0]]['flowRBCTube']
#    		        print(G.es[outEdges[0]]['flowRBCTube'])
#		    #It s a convergent bifurcatin
#    	            elif len(inEdges) > len(outEdges):
#    		        G.es[outEdges[0]]['flowRBCTube']=G.es[inEdges[0]]['flowRBCTube']+G.es[inEdges[1]]['flowRBCTube']
#                        print(G.es[outEdges[0]]['flowRBCTube'])
#		    #There is exactly one in- and one outEdge
#    	            elif len(inEdges) == len(outEdges):
#                        G.es[outEdges[0]]['flowRBCTube']=G.es[inEdges[0]]['flowRBCTube']
#                        print(G.es[outEdges[0]]['flowRBCTube'])
#		    #It s divergent bifurcation
#    	            else:
#			if G.es[outEdges[0]]['flowRBCMax'] == None and G.es[outEdges[1]]['flowRBCMax'] == None:
#			    print(G.es[outEdges[0]]['flow']/G.es[inEdges[0]]['flow'])
#    		            rbcFlowRel=pse(G.es[outEdges[0]]['flow']/G.es[inEdges[0]]['flow'])
#                            print(rbcFlowRel)
#    		            G.es[outEdges[0]]['flowRBCTube']=rbcFlowRel*G.es[inEdges[0]]['flowRBCTube']
#                            G.es[outEdges[1]]['flowRBCTube']=(1-rbcFlowRel)*G.es[inEdges[0]]['flowRBCTube']
#                            print(G.es[outEdges[0]]['flowRBCTube'])
#                        elif G.es[outEdges[0]]['flowRBCMax'] != None and G.es[outEdges[1]]['flowRBCMax'] != None:
#			    G.es[outEdges[0]]['flowRBCTube']=G.es[outEdges[0]]['flowRBCMax']
#                            G.es[outEdges[1]]['flowRBCTube']=G.es[outEdges[1]]['flowRBCMax']
#			else:
#			    if G.es[outEdges[0]]['flowRBCMax'] != None:
#				print('outEdges[0][flowRBCMax] != None')
#				G.es[outEdges[0]]['flowRBCTube']= G.es[outEdges[0]]['flowRBCMax']
#				G.es[outEdges[1]]['flowRBCTube']= G.es[inEdges[0]]['flowRBCTube']-G.es[outEdges[0]]['flowRBCTube']
#			    elif G.es[outEdges[1]]['flowRBCMax'] != None:
#                                print('outEdges[1][flowRBCMax] != None')
#                                G.es[outEdges[1]]['flowRBCTube']= G.es[outEdges[1]]['flowRBCMax']
#                                G.es[outEdges[0]]['flowRBCTube']= G.es[inEdges[0]]['flowRBCTube']-G.es[outEdges[1]]['flowRBCTube']
#		    #Check if tube hematocrit is in the limit for each vessel
#		    for i in range(len(outEdges)):
#		        nRBC=G.es[outEdges[i]]['nRBC']+dt*G.es[outEdges[i]]['flowRBCTube']-\
#			    dt*G.es[outEdges[i]]['htt']*0.25*np.pi*G.es[outEdges[i]]['diameter']**2*G.es[outEdges[i]]['v']/vrbc
#                        htt=nRBC*vrbc/(0.25*np.pi*G.es[outEdges[i]]['diameter']**2*G.es[outEdges[i]]['length'])
#		        if htt > G.es[outEdges[i]]['httMax']:
#			    print('rbcFlow Limit set for vessel')
#			    print(nRBC)
#			    print(htt)
#			    print(G.es[outEdges[i]]['httMax'])
#			    print(G.es[outEdges[i]]['htt'])
#			    print(G.es[outEdges[i]]['nRBC'])
#			    print(G.es[outEdges][i]['v'])
#			    print(outEdges[i])
#			    print(i)
#			    #Calculate maximum rbcFlow for next timestep
#                            nRBCmax=G.es[outEdges[i]]['httMax']*(0.25*np.pi*G.es[outEdges[i]]['diameter']**2*G.es[outEdges[i]]['length'])/vrbc
#			    print(nRBCmax)
#                            flowRBCTubeMax=(nRBCmax-G.es[outEdges[i]]['nRBC']+\
#			        dt*G.es[outEdges[i]]['htt']*0.25*np.pi*G.es[outEdges[i]]['diameter']**2*G.es[outEdges[i]]['v']/vrbc)/dt
#			    print(flowRBCTubeMax)
#			    G.es[outEdges[i]]['flowRBCMax']=flowRBCTubeMax
#                    #There is exactly one in- and one outEdge
#		    if len(inEdges)==len(outEdges) and G.es[outEdges[0]]['flowRBCMax'] != None:
#		        print('There is one in and one outEdge')
#			if G.es[outEdges[0]]['flowRBCMax'] != None and G.es[inEdges[0]]['flowRBCMax'] != None:
#                            print('Limit for inEdge is already set')
#                            print(G.es[inEdges[0]]['flowRBCMax'])
#                            print(G.es[outEdges[0]]['flowRBCMax'])
#                            continue 
#			else:
#                            print('inEdges are limited')
#		            G.es[inEdges[0]]['flowRBCMax']=flowRBCTubeMax
#			    neighbors=G.neighbors(vertex)
#		            vertices=list(set(G.neighbors(neighbors[0])+G.neighbors(neighbors[1])))
#		            pSortedVertices2 = sorted([(v['pressure'], v.index) for v in G.vs[vertices]],reverse=True)
#		            VertexStart=pSortedVertices2[0][1]
#		    #Convergent Bifurcation
#		    elif len(inEdges) > len(outEdges) and G.es[outEdges[0]]['flowRBCMax'] != None:
#                        print('It is a convergent bifurcation')
#			if G.es[inEdges[0]]['flowRBCMax'] != None and G.es[inEdges[1]]['flowRBCMax'] != None:
#			    print('Limit for inEdges is already set')
#			    print(G.es[inEdges[0]]['flowRBCMax'])
#			    print(G.es[inEdges[1]]['flowRBCMax'])
#			    print(G.es[outEdges[0]]['flowRBCMax'])
#			    continue	
#			else:
#			    print('inEdges are limited')
#		            factorRBCin=G.es[inEdges[0]]['flowRBCTube']/G.es[inEdges[1]]['flowRBCTube']
#		            G.es[inEdges[1]]['flowRBCMax']=G.es[outEdges[0]]['flowRBCMax']/(factorRBCin+1)
#		            G.es[inEdges[0]]['flowRBCMax']=G.es[outEdges[0]]['flowRBCMax']-G.es[inEdges[1]]['flowRBCMax']
#                            neighbors=G.neighbors(vertex)
#                            vertices=list(set(G.neighbors(neighbors[0])+G.neighbors(neighbors[1])+G.neighbors(neighbors[2])))
#                            pSortedVertices2 = sorted([(v['pressure'], v.index) for v in G.vs[vertices]],reverse=True)
#                            VertexStart=pSortedVertices2[0][1]
#		    #Inlet	
#		    elif len(inEdges) == 0 and len(outEdges) ==1:
#			continue
#		    #Divergent Bifurcation 
#		    elif len(outEdges) > len(inEdges) and (G.es[outEdges[0]]['flowRBCMax'] != None or G.es[outEdges[1]]['flowRBCMax'] != None):
#                        print('It is a divergent bifurcation')
#			#Both outEdges are full
#		        if G.es[outEdges[0]]['flowRBCMax'] != None and G.es[outEdges[1]]['flowRBCMax'] != None:
#		            print('There are two full outEdges')
#			    if G.es[inEdges[0]]['flowRBCMax'] != None:
#                                print('Limit for inEdges is already set')
#                                print(G.es[inEdges[0]]['flowRBCMax'])
#                                print(G.es[outEdges[0]]['flowRBCMax'])
#                                print(G.es[outEdges[1]]['flowRBCMax'])
#				continue
#			    else:
#			        G.es[inEdges[0]]['flowRBCMax']=G.es[outEdges[0]]['flowRBCMax']+G.es[outEdges[1]]['flowRBCMax']
#                                neighbors=G.neighbors(vertex)
#                                vertices=list(set(G.neighbors(neighbors[0])+G.neighbors(neighbors[1])+G.neighbors(neighbors[2])))
#                                pSortedVertices2 = sorted([(v['pressure'], v.index) for v in G.vs[vertices]],reverse=True)
#                                VertexStart=pSortedVertices2[0][1]
#			        print('rbcFlow Limit set for inEdges at divergent bifurcation')
#			        print(inEdges[0])
#			#Only one outEdge is full
#			else:
#                            print('There is only one full outEdge')
#		            if G.es[outEdges[0]]['flowRBCMax'] != None and G.es[outEdges[1]]['flowRBCMax'] == None:
#			        print('G.es[outEdges[0]][flowRBCMax] != None')
#                                print(G.es[outEdges[0]]['flowRBCMax'])
#			        j=0
#			        k=1
#		            elif G.es[outEdges[1]]['flowRBCMax'] != None and G.es[outEdges[0]]['flowRBCMax'] == None:
#                                print('G.es[outEdges[1]][flowRBCMax] != None')
#                                print(G.es[outEdges[1]]['flowRBCMax'])
#			        k=0
#			        j=1
#		            G.es[outEdges[k]]['flowRBCTube']=G.es[inEdges[0]]['flowRBCTube']-G.es[outEdges[j]]['flowRBCMax']
#                            print(G.es[outEdges[j]]['flowRBCMax'])
#                            print(G.es[outEdges[k]]['flowRBCTube'])
#                            print(G.es[inEdges[0]]['flowRBCTube'])
#		            nRBC=G.es[outEdges[k]]['nRBC']+dt*G.es[outEdges[k]]['flowRBCTube']-\
#                                dt*G.es[outEdges[k]]['htt']*0.25*np.pi*G.es[outEdges[k]]['diameter']**2*G.es[outEdges[k]]['v']/vrbc
#                            htt=G.es[outEdges[k]]['nRBC']*vrbc/(0.25*np.pi*G.es[outEdges[k]]['diameter']**2*G.es[outEdges[k]]['length'])
#		            if htt > G.es[outEdges[k]]['httMax']:
#				print('the maximum for the second outEdge is reached as well')
#                                nRBCmax=G.es[outEdges[k]]['httMax']*(0.25*np.pi*G.es[outEdges[k]]['diameter']**2*G.es[outEdges[k]]['length'])/vrbc
#                                flowRBCTubeMax=(nRBCmax-G.es[outEdges[k]]['nRBC']+\
#                                    dt*G.es[outEdges[k]]['htt']*0.25*np.pi*G.es[outEdges[k]]['diameter']**2*G.es[outEdges[k]]['v']/vrbc)/dt
#                                G.es[outEdges[k]]['flowRBCMax']=flowRBCTubeMax
#		                G.es[inEdges[0]]['flowRBCMax']=G.es[outEdges[k]]['flowRBCMax']+G.es[outEdges[j]]['flowRBCMax']
#                                neighbors=G.neighbors(vertex)
#                                vertices=list(set(G.neighbors(neighbors[0])+G.neighbors(neighbors[1])+G.neighbors(neighbors[2])))
#                                pSortedVertices2 = sorted([(v['pressure'], v.index) for v in G.vs[vertices]],reverse=True)
#                                VertexStart=pSortedVertices2[0][1]
#		            else:
#				G.es[outEdges[j]]['flowRBCTube']=G.es[outEdges[j]]['flowRBCMax']
#			        G.es[outEdges[k]]['flowRBCTube']=G.es[inEdges[0]]['flowRBCTube']-G.es[outEdges[j]]['flowRBCMax']
#	
#		if VertexStart != None:
#		    print('RESTART Flow Distribution at Vertex')
#		    flowMax=[]
#		    print(VertexStart)
#		    print('Edges with flowRBCMax != None')
#		    for i in range(G.ecount()):
#			if G.es[i]['flowRBCMax'] != None:
#			    flowMax.append(i)
#		    print(flowMax)
#		    for i in range(G.vcount()):
#		        if VertexStart == pSortedVertices[i][1]:
#			    line=i
#			    break
#		    pSortedVerticesLoop = pSortedVertices[line::]
#	            break
#
#	    if vertex == pSortedVerticesLoop[-1][1]:
#	       break

    #--------------------------------------------------------------------------

    def _update_hematocrit(self):
        """Computes the new hematocrit values for each edge
        """

	G=self._G
	dt = self._dt
        vrbc = self._P.rbc_volume()
        htt2htd = self._P.tube_to_discharge_hematocrit
	invivo=self._invivo

	#Calculate number of RBCs after dt
	for e in G.es:
            e['nRBC']=e['nRBC']+dt*e['flowRBCIn']-dt*e['htt']*0.25*np.pi*e['diameter']**2*e['v']/vrbc

	#Update tube and discharge hematocrit
	for e in G.es:
	    e['htt']=e['nRBC']*vrbc/(0.25*np.pi*e['diameter']**2*e['length'])
            e['htd']=htt2htd(e['htt'], e['diameter'], invivo)

	    if e['htt'] > 0.99:
		print('WARNING')
		vgm.write_pkl(G,'G_warning.pkl')
		print('htt > 0.99 in vessel')
		print(e.index)

    #--------------------------------------------------------------------------

    def _convergence_check(self):
        """Computes the new hematocrit values for each edge
        """

        sampledict=self._sampledict
	Convergence = self._convergence

	convergence0=0
        convergence1=0
	convergence2=0
        sIndex=len(sampledict['time'])-1
        startChecking=20
	#maxminNRBC=[]
	#maxminFlow=[]
        #maxminRBCFlow=[]
        maxminNRBC=sampledict['nRBCConvergence']
        maxminFlow=sampledict['flowConvergence']
        maxminRBCFlow=sampledict['RBCFlowConvergence']


	print('Check for Convergence')
	if sIndex > startChecking:
	    for j in range(1,startChecking):
		k=startChecking-j
		print('Step of Convergence Checking')
		print(k)
                convergence0=0
                convergence1=0
                convergence2=0
                compareNRBC=100*(np.array(sampledict['nRBC'][sIndex])-np.array(sampledict['nRBC'][sIndex-k]))/np.array(sampledict['nRBC'][sIndex])
                compareFlow=100*(np.array(sampledict['flow'][sIndex])-np.array(sampledict['flow'][sIndex-k]))/np.array(sampledict['flow'][sIndex])
                #compareFlow=100*(np.array(sampledict['sign'][sIndex])*np.array(sampledict['flow'][sIndex])- \
                #    np.array(sampledict['sign'][sIndex-k])*np.array(sampledict['flow'][sIndex-k]))/np.array(sampledict['flow'][sIndex])
                compareRBCFlow=100*(np.array(sampledict['rbcFlow'][sIndex])-np.array(sampledict['rbcFlow'][sIndex-k]))/np.array(sampledict['rbcFlow'][sIndex])
                #compareRBCFlow=100*(np.array(sampledict['sign'][sIndex])*np.array(sampledict['rbcFlow'][sIndex])- \
                #    np.array(sampledict['rbcFlow'][sIndex-k])*np.array(sampledict['sign'][sIndex-k]))/np.array(sampledict['rbcFlow'][sIndex])
		for i in range(len(compareNRBC)):
		    if abs(sampledict['nRBC'][sIndex][i]) < 1e-3 and abs(sampledict['nRBC'][sIndex-k][i]) < 1e-3:
			print('nRBC= 0 in Vessel')
			print(i)
			compareNRBC[i]=0
                    if abs(sampledict['flow'][sIndex][i]) < 1e-3 and abs(sampledict['flow'][sIndex-k][i]) < 1e-3:
                        compareFlow[i]=0
                        print('Flow= 0 in Vessel')
			print(i)
                    if abs(sampledict['rbcFlow'][sIndex][i]) < 1e-3 and abs(sampledict['rbcFlow'][sIndex-k][i]) < 1e-3:
                        print('RBCFlow= 0 in Vessel')
                        print(i)
                        compareRBCFlow[i]=0
                maxminNRBC.append(max(abs(max(compareNRBC)),abs(min(compareNRBC))))
                maxminFlow.append(max(abs(max(compareFlow)),abs(min(compareFlow))))
                maxminRBCFlow.append(max(abs(max(compareRBCFlow)),abs(min(compareRBCFlow))))
                for i in range(len(compareNRBC)):
                    if abs(compareNRBC[i])== maxminNRBC[-1]:
                        print('Maximum nRBC Difference in Vessel')
                        print(i)
			print(compareNRBC[i])
                    if abs(compareFlow[i])== maxminFlow[-1]:
                        print('Maximum Flow Difference in Vessel')
                        print(i)
                        print(compareFlow[i])
                    if abs(compareRBCFlow[i])== maxminRBCFlow[-1]:
                        print('Maximum RBCFlow  Difference in Vessel')
                        print(i)
                        print(compareRBCFlow[i])
	   	convergence0_=[]
		convergence0_Counter=0
    	        for i in range(len(compareNRBC)):
                    print('abs(compareNRBC[i])')
		    print(abs(compareNRBC[i]))
		    print(i)
    		    if abs(compareNRBC[i]) > 5.0:
    		        convergence0=0
			convergence0_Counter += 1
    		    else:
    		        convergence0=1
		    convergence0_.append(convergence0)
		print('Convergence0_Counter')
                print(convergence0_Counter)
		if convergence0_Counter < 3:
		    convergence0=1
		    for k in range(len(convergence0_)):
			if convergence0_[k] == 0:
			    print('Convergence is not reached in vessel')
			    print(k)
		else:
		    break
		print('TOTAL convergence0=')
		print(convergence0)
                convergence1_=[]
                convergence1_Counter=0
                if convergence0 == 1: 
                    for i in range(len(compareNRBC)):
                        if abs(compareFlow[i]) > 5.0:
                            convergence1=0
                            convergence1_Counter += 1
                        else:
                            convergence1=1
                        convergence1_.append(convergence1)
                print('Convergence1_Counter')
                print(convergence1_Counter)
                if convergence1_Counter < 3:
                    convergence1=1
                    for k in range(len(convergence1_)):
                        if convergence1_[k] == 0:
                            print('Convergence is not reached in vessel')
                            print(k)
                else:
                    break
                print('TOTAL convergence1=')
                print(convergence1)
                convergence2_=[]
                convergence2_Counter=0
                if convergence0 == 1 and convergence1 == 1:
                    for i in range(len(compareNRBC)):
                        if abs(compareRBCFlow[i]) > 5.0:
                            convergence2=0
                            convergence2_Counter += 1
                        else:
                            convergence2=1
                        convergence2_.append(convergence2)
                print('Convergence_Counter')
                print(convergence2_Counter)
                if convergence2_Counter < 3:
                    convergence2=1
                    for k in range(len(convergence2_)):
                        if convergence2_[k] == 0:
                            print('Convergence is not reached in vessel')
                            print(k)
                else:
                    break
                print('TOTAL convergence2=')
                print(convergence2)
    	        if convergence0 != 1 or convergence1 != 1 or convergence2 != 1:
		   break
        if convergence0 == 1 and convergence1 == 1 and convergence2 == 1:
	    Convergence = True
	    print('Convergence is reached')
	else:
	    Convergence = False
            print('No Convergence yet')
	
	sampledict['nRBCConvergence']=maxminNRBC
        sampledict['flowConvergence']=maxminFlow
        sampledict['RBCFlowConvergence']=maxminRBCFlow

	self._convergence = Convergence
	self._sampledict = sampledict


