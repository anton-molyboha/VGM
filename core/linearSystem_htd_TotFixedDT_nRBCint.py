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
from pyamg import smoothed_aggregation_solver, rootnode_solver, util
import pyamg
import scipy as sp
from scipy import finfo, ones, zeros
from scipy.sparse import lil_matrix, linalg
from scipy.integrate import quad
from scipy.optimize import root
from .physiology import Physiology
from scipy.sparse.linalg import gmres
from . import units
from . import g_output
from . import vascularGraph
import pdb
from . import run_faster
import time as ttime
import vgm
import logging

__all__ = ['LinearSystemHtdTotFixedDTnRBCint']
# log = vgm.LogDispatcher.create_logger(__name__)
log = logging.getLogger()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class LinearSystemHtdTotFixedDTnRBCint(object):
    """The discrete model is extended such, that a fixed timestep is given. Hence,
    more than one RBC will move per time step.
    It is differentiated between capillaries and larger vessels. At larger Vessels 
    the RBCs are distributed based on the phase separation law. 
    """
    #@profile
    def __init__(self, G, invivo=True,dThreshold=10.0,init=True,**kwargs):
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
               analyzeBifEvents: boolean if bifurcation events should be analyzed (Default = 0)
               innerDiam: boolean if inner or outer diamter of vessels is given in the graph 
                   (innerDiam = 1 --> inner diameter given) (Default = 0)
               analyzeCapDil: edge index of dilated capillary should be given. Upstream divergent bifurcations
                   will be analyzed
               species: 'rat', 'mouse' or 'human', default is 'rat'
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
        self._tPlot = 0.0
        self._tSample = 0.0
        self._filenamelist = []
        self._timelist = []
        #self._filenamelistAvg = []
        self._timelistAvg = []
        self._sampledict = {} 
        #self._transitTimeDict = {}
        self._init=init
        self._scaleToDef=vgm.units.scaling_factor_du('mmHg',G['defaultUnits'])
        self._dtFix=0.0
        self._vertexUpdate=None
        self._edgeUpdate=None
        G.es['source']=[e.source for e in G.es]
        G.es['target']=[e.target for e in G.es]
        G.es['crosssection']=np.array([0.25*np.pi]*G.ecount())*np.array(G.es['diameter'])**2
        G.es['volume']=[e['crosssection']*e['length'] for e in G.es]
        #Used because the Pries functions are onlt defined for vessels till 3micron
        G.es['diamCalcEff']=[i if i >= 3. else 3.0 for i in G.es['diameter'] ]
        G.es['keep_rbcs']=[[] for i in range(G.ecount())]
        adjacent=[]
        self._spacing=[]
        for i in range(G.vcount()):
            adjacent.append(G.adjacent(i))
        G.vs['adjacent']=adjacent

        htd2htt=self._P.discharge_to_tube_hematocrit
        htt2htd = self._P.tube_to_discharge_hematocrit

        if kwargs.has_key('species'):
            self._species = kwargs['species']
        else:
            self._species = 'rat'

        #print('Species')
        #print(self._species)

        if kwargs.has_key('analyzeBifEvents'):
            self._analyzeBifEvents = kwargs['analyzeBifEvents']
        else:
            self._analyzeBifEvents = 0

        if kwargs.has_key('innerDiam'):
            self._innerDiam = kwargs['innerDiam']
        else:
            self._innerDiam = 0

        if kwargs.has_key('analyzeCapDil'):
            self._capDil = kwargs['analyzeCapDil']
            print('capDil')
            print(self._capDil)
            print(G.ecount())
        else:
            self._capDil = 0

        # Assure that both pBC and rBC edge properties are present:
        for key in ['pBC', 'rBC']:
            if not G.vs[0].attributes().has_key(key):
                G.vs[0][key] = None

        if self._analyzeBifEvents:
            self._rbcsMovedPerEdge=[]
            self._edgesWithMovedRBCs=[]
            self._rbcMoveAll=[]
        else:
            if 'rbcMovedAll' in G.attributes():
                del(G['rbcMovedAll'])
            if 'rbcsMovedPerEdge' in G.attributes():
                del(G['rbcsMovedPerEdge'])
            if 'rbcMovedAll' in G.attributes():
                del(G['edgesMovedRBCs'])
        # Set initial pressure and flow to zero:
        if init:
            G.vs['pressure']=zeros(nVertices) 
            G.es['flow']=zeros(G.ecount())    

        G.vs['degree']=G.degree()
        #print('Initial flow, presure, ... assigned')

        if not init:
           if 'averagedCount' not in G.attributes():
               self._sampledict['averagedCount']=0
           else:
               self._sampledict['averagedCount']=G['averagedCount']

        #Calculate total network Volume
        G['V']=0
        for e in G.es:
            G['V']=G['V']+e['crosssection']*e['length']
        #print('Total network volume calculated')

        # Compute the edge-specific minimal RBC distance:
        vrbc = self._P.rbc_volume(self._species)
        if self._innerDiam:
            G.es['minDist'] = [vrbc / (np.pi * e['diameter']**2 / 4) for e in G.es]
        else:
            eslThickness = self._P.esl_thickness
            G.es['minDist'] = [vrbc / (np.pi * (d-2*eslThickness(d))**2 / 4) for d in G.es['diameter']]
        G.es['nMax'] = [np.floor(e['length']/ e['minDist']) for e in G.es] 

        # Assign capillaries and non capillary vertices
        #print('Start assign capillary and non capillary vertices')
        adjacent=[np.array(G.incident(i)) for i in G.vs]
        G.vs['isCap']=[None]*G.vcount()
        self._interfaceVertices=[]
        for i in xrange(G.vcount()):
            #print(i)
            category=[]
            for j in adjacent[i]:
                #print(j)
                #print(G.es.attribute_names())
                if G.es[int(j)]['diameter'] < dThreshold:
                    category.append(1)
                else:
                    category.append(0)
            if category.count(1) == len(category):
                G.vs[i]['isCap']=True
            elif category.count(0) == len(category):
                G.vs[i]['isCap']=False
            else:
                self._interfaceVertices.append(i)
        #print('End assign capillary and non capillary vertices')

        # Arterial-side inflow:
        if init:
            if 'htdBC' in G.es.attribute_names() and 'httBC' not in G.es.attribute_names():
                G.es['httBC']=[e['htdBC'] if e['htdBC'] == None else \
                    self._P.discharge_to_tube_hematocrit(e['htdBC'],e['diameter'],invivo) for e in G.es()]
            if not 'httBC' in G.es.attribute_names():
                for vi in G['av']:
                    for ei in G.adjacent(vi):
                        G.es[ei]['httBC'] = self._P.tube_hematocrit(
                                                G.es[ei]['diameter'], 'a')

        #Convert tube hematocrit boundary condition to htdBC (in case it does not already exist)
        #if not 'htdBC' in G.es.attribute_names():
        #   G.es['htdBC']=[e['httBC'] if e['httBC'] == None else \
        #        self._P.tube_to_discharge_hematocrit(e['httBC'],e['diameter'],invivo) for e in G.es()]
        print('Htt BC assigned')

        httBC_edges = G.es(httBC_ne=None).indices
        #Save initial value of httBC
        if init:
            G.es[httBC_edges]['httBC_init']=G.es[httBC_edges]['httBC']
            httBCValue=np.mean(G.es[httBC_edges]['httBC'])
            for i in G.vs(vv_eq=1).indices:
                if G.es[G.adjacent(i)[0]]['httBC_init'] == None:
                    G.es[G.adjacent(i)[0]]['httBC_init']=httBCValue

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
            #in every edge with httBC should be at least on RBC in the initial state. This is necessary to
            #properly initialize v_last
            httBC_edges = G.es(httBC_ne=None)
            for e in httBC_edges:
                if len(e['rRBC']) == 0:
                    Nmax = max(int(np.floor(e['nMax'])), 1)
                    N=1
                    indices = sorted(np.random.permutation(Nmax)[:N])
                    e['rRBC'] = np.array(indices) * lrbc + lrbc / 2.0
         
        G.es['nRBC']=[len(e['rRBC']) for e in G.es]
        G.es['nRBCint']=G.es['nRBC']

        if kwargs.has_key('plasmaViscosity'):
            self._muPlasma = kwargs['plasmaViscosity']
        else:
            self._muPlasma = self._P.dynamic_plasma_viscosity()

        # Compute nominal and specific resistance:
        self._update_nominal_and_specific_resistance()
        #print('Resistance updated')

        # Compute the current tube hematocrit from the RBC positions:
        for e in G.es:
            e['htt']=min(len(e['rRBC'])*vrbc/e['volume'],1)
            e['htd']=min(htt2htd(e['htt'], e['diameter'], invivo), 1.0)
        #print('Initial htt and htd computed')        

        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']*self._scaleToDef
        self._update_eff_resistance_and_LS(None, None)
        #print('Matrix created')
        self._solve('iterative2')
        #print('Matrix solved')
        self._G.vs['pressure'] = deepcopy(self._x)
        #Convert deaultUnits to 'pBC' ['mmHG']
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']/self._scaleToDef
        self._update_flow_and_velocity()
        #print('Flow updated')
        self._update_flow_sign()
        #print('Flow sign updated')
        self._update_out_and_inflows_for_vertices()
        #print('updated out and inflows')

        eps=self._eps
        if init:
            G.es['posFirst_last']=[None]*G.ecount()
            G.es['logNormal']=[None]*G.ecount()
            httBC_edges = G.es(httBC_ne=None)
            #print('Update logNormal')
            #print(len(httBCInit_edges))
            #print(G.ecount())
            for e in httBC_edges:
                if len(e['rRBC']) > 0:
                    if e['sign'] == 1:
                        e['posFirst_last']=e['rRBC'][0]
                    else:
                        e['posFirst_last']=e['length']-e['rRBC'][-1]
                else:
                    e['posFirst_last']=e['length']
                e['v_last']=0
                httBCValue=e['httBC_init']
                if self._innerDiam:
                    LDValue = httBCValue
                else:
                    LDValue=httBCValue*(e['diameter']/(e['diameter']-2*eslThickness(e['diameter'])))**2
                logNormalMu,logNormalSigma=self._compute_mu_sigma_inlet_RBC_distribution(LDValue)
                #print('Initial nRBC computed')    
                e['logNormal']=[logNormalMu,logNormalSigma]

                #prepare httBC edges 
                rRBC = []
                lrbc = e['minDist']
                htt = e['httBC']
                length = e['length']
                nMaxNew=e['nMax']-len(e['rRBC'])
                if len(e['rRBC']) > 0:
                    posFirst=e['rRBC'][0] if e['sign']==1.0 else e['length']-e['rRBC'][-1]
                    e['posFirst_last']=posFirst
                    e['v_last'] = e['v']
                    cum_length = posFirst
                else:
                    print('ERROR this should not happen')
                    cum_length = e['posFirst_last']
                    posFirst = cum_length
                    e['posFirst_last']=posFirst
                    #e['v_last']=e['v']
                while cum_length >= lrbc and nMaxNew > 0:
                    if len(e['keep_rbcs']) != 0:
                        if posFirst - e['keep_rbcs'][0] >= 0:
                            if posFirst - e['keep_rbcs'][0] > e['length']:
                                rRBC.append(e['length'])
                                posFirst=e['length']
                            else:
                                rRBC.append(posFirst - e['keep_rbcs'][0])
                                posFirst=posFirst - e['keep_rbcs'][0]
                            nMaxNew += -1
                            cum_length = posFirst
                            e['keep_rbcs']=[]
                            e['posFirst_last']=posFirst
                            #e['v_last']=e['v']
                        else:
                            break
                    else:
                        #number of RBCs randomly chosen to average htt
                        number=np.exp(e['logNormal'][0]+e['logNormal'][1]*np.random.randn(1)[0])
                        #self._spacing.append(number)
                        spacing = lrbc+lrbc*number
                        if posFirst - spacing >= 0:
                            if posFirst - spacing > e['length']:
                                rRBC.append(e['length'])
                                posFirst=e['length']
                            else:
                                rRBC.append(posFirst - spacing)
                                posFirst=posFirst - spacing
                            nMaxNew += -1
                            cum_length = posFirst
                            e['posFirst_last']=posFirst
                            #e['v_last']=e['v']
                        else:
                            e['keep_rbcs']=[spacing]
                            #e['v_last']=e['v']
                            if len(rRBC) == 0:
                                e['posFirst_last']=posFirst
                            else:
                                e['posFirst_last']=rRBC[-1]
                            break
                if len(e['keep_rbcs']) == 0:
                    number=np.exp(e['logNormal'][0]+e['logNormal'][1]*np.random.randn(1)[0])
                    spacing = lrbc+lrbc*number
                    e['keep_rbcs']=[spacing]
                rRBC = np.array(rRBC)
                if len(rRBC) >= 1.:
                    if e['sign'] == 1:
                        e['rRBC'] = np.concatenate([rRBC[::-1], e['rRBC']])
                    else:
                        rRBC=length-rRBC
                        e['rRBC'] = np.concatenate([e['rRBC'], rRBC])
                if len(e['rRBC']) > 0:
                    if e['rRBC'][0] > e['rRBC'][-1] + eps:
                        print('BIGERROR BEGINNING END')
                        print(e.index)
                        print(e['rRBC'][0])
                        print(e['sign'])
                        print(e['rRBC'][-1])
                        print(rRBC)
                    if e['rRBC'][0] < 0 - eps or e['rRBC'][-1] > e['length'] + eps:
                        print('BIGERROR BEGINNING END 2')
                        print(e.index)
                        print(e['rRBC'][0])
                        print(e['rRBC'][-1])
                        print(e['rRBC'])
                        print(e['length'])
                        print(e['sign'])
                    for j in range(len(e['rRBC'])-1):
                       if e['rRBC'][j] > e['rRBC'][j+1] or e['rRBC'][j+1]-e['rRBC'][j] < e['minDist']-100*eps:
                            print('BIGERROR BEGINNING END 3')
                            print(e.index)
                            print(e['rRBC'][j])
                            print(e['rRBC'][j+1])
                            print(e['rRBC'][j+1]-e['rRBC'][j])
                            print(e['minDist'])
                            print(e['sign'])
        #print('Initiallize posFirst_last')
        G.es['nRBC']=[len(e['rRBC']) for e in G.es]
        G.es['nRBCint']=G.es['nRBC']

        # Compute the current tube hematocrit from the RBC positions:
        for e in G.es:
            e['htt']=min(len(e['rRBC'])*vrbc/e['volume'],1)
            e['htd']=min(htt2htd(e['htt'], e['diameter'], invivo), 1.0)
        #print('Initial htt and htd computed')        

        # This initializes the full LS. Later, only relevant parts of
        # the LS need to be changed at any timestep. Also, RBCs are
        # removed from no-flow edges to avoid wasting computational
        # time on non-functional vascular branches / fragments:
        #Convert 'pBC' ['mmHG'] to default Units
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']*self._scaleToDef
        self._update_eff_resistance_and_LS(None, None)
        #print('Matrix created')
        self._solve('iterative2')
        #print('Matrix solved')
        self._G.vs['pressure'] = deepcopy(self._x)
        #Convert deaultUnits to 'pBC' ['mmHG']
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']/self._scaleToDef
        self._update_flow_and_velocity()
        #print('Flow updated')
        self._verify_mass_balance()
        #print('Mass balance verified updated')
        self._update_flow_sign()
        #print('Flow sign updated')
        self._update_out_and_inflows_for_vertices()
        #print('updated out and inflows')

        Gdummy=deepcopy(G)
        #Find upstream divergent bifurcations
        if self._capDil != 0:
            Gdummy.to_directed_flow_based()
            e=Gdummy.es[self._capDil]
            self._upstreamDivBifs=[]
            self._upstreamAFeedBifs=[]
            upstreamBifs=[]
            paths=[]
            newPaths=[]
            direction='in'
            upstreamBifs.append(e.source)
            if Gdummy.vs['vType'][e.source]== 3:
                self._upstreamDivBifs.append(e.source)
            paths.append(upstreamBifs)
            countDone = 0
            #print('Start at')
            #print(paths)
            #print(upstreamBifs)
            #print(self._upstreamDivBifs)
            while len(self._upstreamDivBifs) < 3 and len(paths) != countDone:
                countDone = 0
                for path in paths:
                    #print('paths')
                    #print(path)
                    #print(self._upstreamDivBifs)
                    #print(G.vs[self._upstreamDivBifs]['vType'])
                    #print(Gdummy.neighbors(path[-1],type=direction))
                    if Gdummy.neighbors(path[-1],type=direction) == []:
                        newPaths.append(path)
                        countDone += 1
                    else:
                        for neighbor in Gdummy.neighbors(path[-1],type=direction):
                            #if neighbor not in path:
                            newPaths.append(path + [neighbor])
                            if Gdummy.vs['vType'][neighbor] == 3:
                                self._upstreamDivBifs.append(neighbor)
                            if Gdummy.vs['kind'][neighbor] == 'a':
                                self._upstreamAFeedBifs.append(neighbor)
                                countDone += 1
                paths = newPaths
                newPaths = []
                if countDone == len(paths):
                    #print('countDone = len(paths)')
                    break
            #print('Check Feeding Arteries')
            #print(self._upstreamAFeedBifs)
            self._analyzeDivBifEvents=[]
            self._DivBifEdges=[]
            self._upstreamDivBifsCount = []
            self._upstreamDivBifsNotDiv = []
            for i in self._upstreamDivBifs:
                self._upstreamDivBifsCount.append(0)
                self._upstreamDivBifsNotDiv.append(0)
                dummy=[]
                dummy2=[]
                for j in Gdummy.adjacent(i,'out'):
                    dummy.append(j)
                    dummy2.append(0)
                self._DivBifEdges.append(dummy)
                self._analyzeDivBifEvents.append(dummy2)
            #print('upstreamDivBifs')
            #print(self._upstreamDivBifs)
            #print(self._DivBifEdges)
             
        del(Gdummy)
        #Calculate an estimated network turnover time (based on conditions at the beginning)
        flowsum=0

        for vi in G['av']:
            for ei in G.adjacent(vi):
                flowsum=flowsum+G.es['flow'][ei]
        G['flowSumIn']=flowsum
        G['Ttau']=G['V']/flowsum
        #print(flowsum)
        #print(G['V'])
        #print(self._eps)
        stdout.write("\rEstimated network turnover time Ttau=%f        \n" % G['Ttau'])

        #for e in self._G.es(flow_le=self._eps*1e6):
        #    e['rRBC'] = []
            
    #--------------------------------------------------------------------------

    def _compute_mu_sigma_inlet_RBC_distribution(self, httBC):
        """Updates the nominal and specific resistance of a given edge 
        sequence.
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge properties 'resistance' and 'specificResistance'
                are updated (or created).
        """
        #mean_LD=0.28
        mean_LD=httBC
        std_LD=0.1*mean_LD
        if std_LD > 0.01:
            std_LD = 0.01
        #std_LD=0.1
        #if std_LD > mean_LD:
        #    std_LD = mean_LD
        
        #PDF log-normal
        f_x = lambda x,mu,sigma: 1./(x*np.sqrt(2*np.pi)*sigma)*np.exp(-1*(np.log(x)-mu)**2/(2*sigma**2))
        
        #PDF log-normal for line density
        f_LD = lambda z,mu,sigma: 1./((z-z**2)*np.sqrt(2*np.pi)*sigma)*np.exp(-1*(np.log(1./z-1)-mu)**2/(2*sigma**2))
        
        #f_mean integral dummy
        f_mean_LD_dummy = lambda z,mu,sigma: z*f_LD(z,mu,sigma)
        
        ##calculate mean
        f_mean_LD = lambda mu,sigma: quad(f_mean_LD_dummy,0,1,args=(mu,sigma))[0]
        f_mean_LD_Calc=np.vectorize(f_mean_LD)
        
        #f_var integral dummy
        f_var_LD_dummy = lambda z,mu,sigma: (z-mean_LD)**2*f_LD(z,mu,sigma)
        
        #calculate mean
        f_var_LD = lambda mu,sigma: quad(f_var_LD_dummy,0,1,args=(mu,sigma))[0]
        f_var_LD_Calc=np.vectorize(f_var_LD)
        
        #Set up system of equations
        def f_moments_LD(m):
            x,y=m
            return (f_mean_LD_Calc(x,y)-mean_LD,f_var_LD_Calc(x,y)-std_LD**2)

        optionsSolve={}
        optionsSolve['xtol']=1e-20
        if mean_LD < 0.35:
            sol=root(f_moments_LD,(0.89,0.5),method='lm',options=optionsSolve)
        elif mean_LD > 0.63:
            sol=root(f_moments_LD,(-0.6,0.45),method='lm',options=optionsSolve)
        else:
            sol=root(f_moments_LD,(mean_LD,std_LD),method='lm',options=optionsSolve)
        mu=sol['x'][0]
        sigma=sol['x'][1]

        return mu,sigma

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

    def _update_hematocrit(self, esequence=None):
        """Updates the tube hematocrit of a given edge sequence.
        INPUT: es: Sequence of edge indices as tuple. If not provided, all 
                   edges are updated.
        OUTPUT: None, the edge property 'htt' is updated (or created).
        """
        G = self._G
        htt2htd = self._P.tube_to_discharge_hematocrit
        invivo=self._invivo
        vrbc = self._P.rbc_volume(self._species)

        if esequence is None:
            es = G.es
        else:
            es = G.es(esequence)

        es['htt'] = [min(e['nRBCint'] * vrbc / e['volume'],1) for e in es]
        es['htd']= [min(htt2htd(e['htt'], e['diameter'], invivo), 1.0) for e in es]

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
        if 'sign' in G.es.attributes() and None not in G.es['sign']:
            G.es['signOld']=G.es['sign']
        G.es['sign'] = [np.sign(G.vs[source]['pressure'] -
                                G.vs[target]['pressure']) for source,target in zip(G.es['source'],G.es['target'])]

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
        divergentV=[]
        convergentV=[]
        connectingV=[]
        doubleConnectingV=[]
        noFlowV=[]
        noFlowE=[]
        vertices=[]
        dThreshold = self._dThreshold
        count=0
        interfaceVertices=self._interfaceVertices
        #print('In update out and inflows')
        if not 'sign' in G.es.attributes() or not 'signOld' in G.es.attributes():
            #print('Initial vType Update')
            for v in G.vs:
                vI=v.index
                outE=[]
                inE=[]
                noFlowE=[]
                pressure = G.vs[vI]['pressure']
                adjacents=G.adjacent(vI)
                for j,nI in enumerate(G.neighbors(vI)):
                    #outEdge
                    if pressure > G.vs[nI]['pressure']: 
                        outE.append(adjacents[j])
                    elif pressure == G.vs[nI]['pressure']: 
                        noFlowE.append(adjacents[j])
                    #inflowEdge
                    else: #G.vs[vI]['pressure'] < G.vs[nI]['pressure']
                        inE.append(adjacents[j])
                        #Deal with vertices at the interface
                        #isCap is defined based on the diameter of the InflowEdge
                        if vI in interfaceVertices and G.es[adjacents[j]]['diameter'] > dThreshold:
                            G.vs[vI]['isCap']=False
                        else:
                            G.vs[vI]['isCap']=True
                #Group into divergent, convergent and connecting Vertices
                if len(outE) > len(inE) and len(inE) >= 1:
                    divergentV.append(vI)
                elif len(inE) > len(outE) and len(outE) >= 1:
                    convergentV.append(vI)
                elif len(inE) == len(outE) and len(inE) == 1:
                    connectingV.append(vI)
                elif len(inE) == len(outE) and len(inE) == 2:
                    doubleConnectingV.append(vI)
                elif vI in G['av']:
                    if len(inE) == 0 and len(outE) == 1:
                        pass
                    elif len(inE) == 1 and len(outE) == 0:
                        #print('WARNING1 boundary condition changed: from av --> vv')
                        #print(vI)
                        G.vs[vI]['av'] = 0
                        G.vs[vI]['vv'] = 1
                        G.vs[vI]['vType'] = 2
                        edgeVI=G.adjacent(vI)[0]
                        G.es[edgeVI]['httBC']=None
                        G.es[edgeVI]['posFirst_last']=None
                        G.es[edgeVI]['v_last']=None
                        #print(G.es[edgeVI]['v_last'])
                    elif len(inE) == 0 and len(outE) == 0:
                        #print('WARNING changed to noFlow edge')
                        edgeVI=G.adjacent(vI)[0]
                        noFlowV.append(vI)
                        noFlowE.append(edgeVI)
                    #else:
                        #print('ERROR in defining in and outlets')
                        #print(vI)
                elif vI in G['vv']:
                    if len(inE) == 1 and len(outE) == 0:
                        pass
                    elif len(inE) == 0 and len(outE) == 1:
                        #print('WARNING1 boundary condition changed: from vv --> av')
                        #print(vI)
                        G.vs[vI]['av'] = 1
                        G.vs[vI]['vv'] = 0
                        G.vs[vI]['vType'] = 1
                        edgeVI=G.adjacent(vI)[0]
                        G.es[edgeVI]['httBC']=G.es[edgeVI]['httBC_init']
                        if len(G.es[edgeVI]['rRBC']) > 0:
                            if G.es['sign'][edgeVI] == 1:
                                G.es[edgeVI]['posFirst_last']=G.es['rRBC'][edgeVI][0]
                            else:
                                G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]-G.es['rRBC'][edgeVI][-1]
                        else:
                            G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]
                        G.es[edgeVI]['v_last']=G.es['v'][edgeVI]
                        #print(G.es[edgeVI]['v_last'])
                    elif len(inE) == 0 and len(outE) == 0:
                        #print('WARNING changed to noFlow edge')
                        noFlowV.append(vI)
                        noFlowE.append(edgeVI)
                    #else:
                        #print('ERROR in defining in and outlets')
                        #print(vI)
                else:
                    for i in G.adjacent(vI):
                        #if G.es['flow'][i] > 5.0e-08:
                            #print('FLOWERROR')
                            #print(vI)
                            #print(inE)
                            #print(outE)
                            #print(noFlowE)
                            #print(i)
                            #print('Flow and diameter')
                            #print(G.es['flow'][i])
                            #print(G.es['diameter'][i])
                        noFlowE.append(i)
                    inE=[]
                    outE=[]
                    noFlowV.append(vI)
                    #print('noFlow V')
                    #print(vI)
                inEdges.append(inE)
                outEdges.append(outE)
            G.vs['inflowE']=inEdges
            G.vs['outflowE']=outEdges
            G.es['noFlow']=[0]*G.ecount()
            noFlowE=np.unique(noFlowE)
            G.es[noFlowE]['noFlow']=[1]*len(noFlowE)
            G['divV']=divergentV
            G['conV']=convergentV
            G['connectV']=connectingV
            G['dConnectV']=doubleConnectingV
            G['noFlowV']=noFlowV
            #print('assign vertex types')
            #vertex type av = 1, vv = 2,divV = 3, conV = 4, connectV = 5, dConnectV = 6, noFlowV = 7
            G.vs['vType']=[0]*G.vcount()
            G['av']=G.vs(av_eq=1).indices
            G['vv']=G.vs(vv_eq=1).indices
            for i in G['av']:
                G.vs[i]['vType']=1
            for i in G['vv']:
                G.vs[i]['vType']=2
            for i in G['divV']:
                G.vs[i]['vType']=3
            for i in G['conV']:
                G.vs[i]['vType']=4
            for i in G['connectV']:
                G.vs[i]['vType']=5
            for i in G['dConnectV']:
                G.vs[i]['vType']=6
            for i in G['noFlowV']:
                G.vs[i]['vType']=7
            #if len(G.vs(vType_eq=0).indices) > 0:
                #print('BIGERROR vertex type not assigned')
                #print(len(G.vs(vType_eq=0).indices))
            del(G['divV'])
            del(G['conV'])
            del(G['connectV'])
            del(G['dConnectV'])
        #Every Time Step
        else:
            #print('Vtype reupdated')
            #print('Update_Out_and_inflows')
            if G.es['sign']!=G.es['signOld']:
                sign=np.array(G.es['sign'])
                signOld=np.array(G.es['signOld'])
                sumTes=abs(sign+signOld)
                #find edges where sign change took place
                edgeList=np.array(np.where(sumTes < abs(2))[0])
                edgeList=edgeList.tolist()
                sign0=G.es(sign_eq=0,signOld_eq=0).indices
                for e in sign0:
                    edgeList.remove(e)
                stdout.flush()
                vertices=[]
                for e in edgeList:
                    for vI in G.es[int(e)].tuple:
                        vertices.append(vI)
                vertices=np.unique(vertices)
                count = 0
                for vI in vertices:
                    #vI=v.index
                    count += 1
                    vI=int(vI)
                    outE=[]
                    inE=[]
                    noFlowE=[]
                    neighbors=G.neighbors(vI)
                    pressure = G.vs[vI]['pressure']
                    adjacents=G.adjacent(vI)
                    for j in range(len(neighbors)):
                        nI=neighbors[j]
                        #outEdge
                        if pressure > G.vs[nI]['pressure']:
                            outE.append(adjacents[j])
                        elif pressure == G.vs[nI]['pressure']:
                            noFlowE.append(adjacents[j])
                        #inflowEdge
                        else: #G.vs[vI]['pressure'] < G.vs[nI]['pressure']
                            inE.append(adjacents[j])
                            #Deal with vertices at the interface
                            #isCap is defined based on the diameter of the InflowEdge
                            if vI in interfaceVertices and G.es[adjacents[j]]['diameter'] > dThreshold:
                                G.vs[vI]['isCap']=False
                            else:
                                G.vs[vI]['isCap']=True
                    #Group into divergent, convergent, connecting, doubleConnecting and noFlow Vertices
                    #it is now a divergent Vertex
                    if len(outE) > len(inE) and len(inE) >= 1:
                        #Find history of vertex
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        G.vs[vI]['vType']=3
                        G.vs[vI]['inflowE']=inE
                        G.vs[vI]['outflowE']=outE
                    #it is now a convergent Vertex
                    elif len(inE) > len(outE) and len(outE) >= 1:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        G.vs[vI]['vType']=4
                        G.vs[vI]['inflowE']=inE
                        G.vs[vI]['outflowE']=outE
                    #it is now a connecting Vertex
                    elif len(outE) == len(inE) and len(outE) == 1:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        G.vs[vI]['vType']=5
                        G.vs[vI]['inflowE']=inE
                        G.vs[vI]['outflowE']=outE
                    #it is now a double connecting Vertex
                    elif len(outE) == len(inE) and len(outE) == 2:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        G.vs[vI]['vType']=6
                        G.vs[vI]['inflowE']=inE
                        G.vs[vI]['outflowE']=outE
                    elif vI in G['av']:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        if G.vs[vI]['rBC'] != None:
                            for j in G.adjacent(vI):
                                if G.es[j]['flow'] > 1e-6:
                                    #print(' ')
                                    #print(vI)
                                    #print(len(G.vs[vI]['inflowE']))
                                    #print(len(G.vs[vI]['outflowE']))
                                    print('ERROR flow direction of inlet vertex changed')
                                    #print(G.es[G.adjacent(vI)]['flow'])
                                    #print(G.vs[vI]['rBC'])
                                    #print(G.vs[vI]['kind'])
                                    #print(G.vs[vI]['isSrxtm'])
                                    #print(G.es[G.adjacent(vI)]['sign'])
                                    #print(G.es[G.adjacent(vI)]['signOld'])
                        else:
                            print('WARNING direction out av changed to vv')
                            print(vI)
                            G.vs[vI]['av'] = 0
                            G.vs[vI]['vv'] = 1
                            G.vs[vI]['vType'] = 2
                            edgeVI=G.adjacent(vI)[0]
                            G.es[edgeVI]['httBC']=None
                            G.es[edgeVI]['posFirst_last']=None
                            G.es[edgeVI]['v_last']=None
                            G.vs[vI]['inflowE']=inE
                            G.vs[vI]['outflowE']=outE
                    elif vI in G['vv']:
                        if G.vs[vI]['vType']==7:
                            G.es[inE]['noFlow']=[0]*len(inE)
                            G.es[outE]['noFlow']=[0]*len(outE)
                        if G.vs[vI]['rBC'] != None:
                            for j in G.adjacent(vI):
                                if G.es[j]['flow'] > 1e-6:
                                    #print(' ')
                                    #print(vI)
                                    #print(len(G.vs[vI]['inflowE']))
                                    #print(len(G.vs[vI]['outflowE']))
                                    print('ERROR flow direction of out vertex changed')
                                    #print(G.es[G.adjacent(vI)]['flow'])
                                    #print(G.vs[vI]['rBC'])
                                    #print(G.vs[vI]['kind'])
                                    #print(G.vs[vI]['isSrxtm'])
                                    #print(G.es[G.adjacent(vI)]['sign'])
                                    #print(G.es[G.adjacent(vI)]['signOld'])
                        else:
                            print('WARNING direction out vv changed to av')
                            print(vI)
                            G.vs[vI]['av'] = 1
                            G.vs[vI]['vv'] = 0
                            G.vs[vI]['vType'] = 1
                            edgeVI=G.adjacent(vI)[0]
                            G.es[edgeVI]['httBC']=G.es[edgeVI]['httBC_init']
                            if len(G.es[edgeVI]['rRBC']) > 0:
                                if G.es['sign'][edgeVI] == 1:
                                    G.es[edgeVI]['posFirst_last']=G.es['rRBC'][edgeVI][0]
                                else:
                                    G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]-G.es['rRBC'][edgeVI][-1]
                            else:
                                G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]
                            G.es[edgeVI]['v_last']=G.es[edgeVI]['v']
                            G.vs[vI]['inflowE']=inE
                            G.vs[vI]['outflowE']=outE
                            if G.es[edgeVI]['logNormal'] == None:
                                httBCValue=G.es[edgeVI]['httBC_init']
                                if self._innerDiam:
                                    LDValue = httBCValue
                                else:
                                    LDValue=httBCValue*(G.es[edgeVI]['diameter']/(G.es[edgeVI]['diameter']-2*eslThickness(G.es[edgeVI]['diameter'])))**2
                                logNormalMu,logNormalSigma=self._compute_mu_sigma_inlet_RBC_distribution(LDValue)
                                G.es[edgeVI]['logNormal']=[logNormalMu,logNormalSigma]
                    #it is now a noFlow Vertex
                    else:
                        if G.vs[vI]['degree']==1 and len(inE) == 1 and len(outE) == 0:
                            #print('WARNING2 changed from noFlow to outflow')
                            #print(vI)
                            G.vs[vI]['av'] = 0
                            G.vs[vI]['vv'] = 1
                            G.vs[vI]['vType'] = 2
                            edgeVI=G.adjacent(vI)[0]
                            G.es[edgeVI]['httBC']=None
                            G.es[edgeVI]['posFirst_last']=None
                            G.es[edgeVI]['v_last']=None
                        elif G.vs[vI]['degree']==1 and len(inE) == 0 and len(outE) == 1:
                            #print('WARNING2 changed from noFlow to inflow')
                            #print(vI)
                            G.vs[vI]['av'] = 1
                            G.vs[vI]['vv'] = 0
                            G.vs[vI]['vType'] = 1
                            edgeVI=G.adjacent(vI)[0]
                            G.es[edgeVI]['httBC']=G.es[edgeVI]['httBC_init']
                            if len(G.es[edgeVI]['rRBC']) > 0:
                                if G.es['sign'][edgeVI] == 1:
                                    G.es[edgeVI]['posFirst_last']=G.es['rRBC'][edgeVI][0]
                                else:
                                    G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]-G.es['rRBC'][edgeVI][-1]
                            else:
                                G.es[edgeVI]['posFirst_last']=G.es['length'][edgeVI]
                            G.es[edgeVI]['v_last']=G.es[edgeVI]['v']
                        else:
                            noFlowEdges=[]
                            for i in G.adjacent(vI):
                                #if G.es['flow'][i] > 5.0e-08:
                                    #print('FLOWERROR')
                                    #print(vI)
                                    #print(inE)
                                    #print(outE)
                                    #print(noFlowE)
                                    #print(i)
                                    #print('Flow and diameter')
                                    #print(G.es['flow'][i])
                                    #print(G.es['diameter'][i])
                                noFlowEdges.append(i)
                            G.vs[vI]['vType']=7
                            G.es[noFlowEdges]['noFlow']=[1]*len(noFlowEdges)
                            G.vs[vI]['inflowE']=[]
                            G.vs[vI]['outflowE']=[]

            G['av']=G.vs(av_eq=1).indices
            G['vv']=G.vs(vv_eq=1).indices
        stdout.flush()
        if len(G.vs(av_eq=1,degree_gt=1))>0:
            #print('BIGERROR av')
            G['avProb']=G.vs(av_eq=1,degree_gt=1).indices
            vgm.write_pkl(G,'Gavprob.pkl') 
        if len(G.vs(vv_eq=1,degree_gt=1))>0:
            #print('BIGERROR vv')
            G['vvProb']=G.vs(vv_eq=1,degree_gt=1).indices
            vgm.write_pkl(G,'Gvvprob.pkl') 

    #--------------------------------------------------------------------------

    def _update_flow_and_velocity(self):
        """Updates the flow and red blood cell velocity in all vessels
        INPUT: None
        OUTPUT: None
        """

        G = self._G
        invivo=self._invivo
        vf = self._P.velocity_factor
        vrbc = self._P.rbc_volume(self._species)
        vfList=[1.0 if htt == 0.0 else max(1.0,vf(d, invivo, tube_ht=htt)) for d,htt in zip(G.es['diameter'],G.es['htt'])]

        self._G=run_faster.update_flow_and_v(self._G,self._invivo,vfList,vrbc)
        G= self._G

        #G = self._G
        #invivo=self._invivo
        #vf = self._P.velocity_factor
        #pi=np.pi
        #G.es['flow'] = np.array([abs(G.vs[e.source]['pressure'] -                                           
        #                    G.vs[e.target]['pressure']) /res                        
        #                    for e,res in zip(G.es,G.es['effResistance'])])
        # RBC velocity is not defined if tube_ht==0, using plasma velocity
        # instead:
        #G.es['v'] = [4 * flow * vf(d, invivo, tube_ht=htt) /                  
        #            (pi * d**2) if htt > 0 else                                
        #            4 * flow / (pi * d**2)                                     
         #           for flow,d,htt in zip(G.es['flow'],G.es['diameter'],G.es['htt'])]

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
            vertexList=[]
            edgeList=[]
            for i in vertex:
                vList = np.concatenate([[i],
                     G.neighbors(i)]).tolist()
                eList = G.adjacent(i)
                vertexList=np.concatenate([vertexList,vList]).tolist()
                edgeList=np.concatenate([edgeList,eList]).tolist()
            vertexList=np.unique(vertexList).tolist()
            edgeList=np.unique(edgeList).tolist()
            edgeList=[int(i) for i in edgeList]
            vertexList=[int(i) for i in vertexList]
        dischargeHt = [min(htt2htd(e, d, invivo), 1.0) for e,d in zip(G.es[edgeList]['htt'],G.es[edgeList]['diameter'])]
        G.es[edgeList]['effResistance'] =[ res * nurel(max(d,4.0), min(dHt,0.6),invivo) for res,dHt,d in zip(G.es[edgeList]['resistance'], \
            dischargeHt,G.es[edgeList]['diamCalcEff'])]

        edgeList = G.es(edgeList)
        vertexList = G.vs(vertexList)
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

        self._A = A
        self._b = b
        self._G = G

    #--------------------------------------------------------------------------
    #@profile
    def _propagate_rbc(self):
        """This assigns the current bifurcation-RBC to a new edge and
        propagates all RBCs until the next RBC reaches at a bifurcation.
        INPUT: None
        OUTPUT: None
        """
        G = self._G
        dt = self._dt # Time to propagate RBCs with current velocity.
        eps=self._eps
        Physiol = self._P
        if self._capDil != 0:
            upstreamDivBifs = self._upstreamDivBifs
            divBifEdges=self._DivBifEdges
            analyzeDivBifEvents=self._analyzeDivBifEvents
            upstreamDivBifsCount=self._upstreamDivBifsCount
            upstreamDivBifsNotDiv=self._upstreamDivBifsNotDiv
        #No flow Edges are not considered for the propagation of RBCs
        edgeList0=G.es(noFlow_eq=0).indices
        if self._analyzeBifEvents:
            rbcsMovedPerEdge=[]
            edgesWithMovedRBCs=[]
            rbcMoved = 0
        edgeList=G.es[edgeList0]
        #Edges are sorted based on the pressure at the outlet
        pOut=[G.vs[e['target']]['pressure'] if e['sign'] == 1.0 else G.vs[e['source']]['pressure']
            for e in edgeList]
        sortedE=zip(pOut,edgeList0)
        sortedE.sort()
        sortedE=[i[1] for i in sortedE]
        convEdges2=[]
        edgeUpdate=[]   #Edges where the number of RBCs changed --> need to be updated
        vertexUpdate=[] #Vertices where the number of RBCs changed in adjacent edges --> need to be updated
        G.es['nRBCint']=[0]*G.ecount()
        #SECOND step go through all edges from smallest to highest pressure and move RBCs
        for ei in sortedE:
            noBifEvents = 0
            edgesInvolved=[] #all edges connected to the bifurcation vertex
            e = G.es[ei]
            sign=e['sign']
            #Get bifurcation vertex
            if sign == 1:
                vi=e.target
            else:
                vi=e.source
            for i in G.vs[vi]['inflowE']:
                 edgesInvolved.append(i)
            for i in G.vs[vi]['outflowE']:
                 edgesInvolved.append(i)
            nRBCSumBefore=0
            for i in edgesInvolved:
                nRBCSumBefore += G.es[i]['nRBC']
            overshootsNo=0 #Reset - Number of overshoots acutally taking place (considers possible number of bifurcation events)
            #If there is a BC for the edge new RBCs have to be introduced
            #Check if the RBCs in the edge have been moved already (--> convergent bifurcation)
            #Recheck if bifurcation vertex is a noFlow Vertex (vType=7)
            boolHttEdge = 0
            boolHttEdge2 = 0
            boolHttEdge3 = 0
            if G.vs[vi]['vType'] == 7:
                e['nRBCint']=len(e['rRBC'])
            if ei not in convEdges2 and G.vs[vi]['vType'] != 7:
            #Check if the RBCs in the edge have been moved already (--> convergent bifurcation)
            #Recheck if bifurcation vertex is a noFlow Vertex (vType=7)
            #if ei not in convEdges2 and G.vs[vi]['vType'] != 7:
                #If RBCs are present move all RBCs
                if len(e['rRBC']) > 0:
                    rRBCold=e['rRBC']
                    e['rRBC'] = e['rRBC'] + e['v'] * dt * e['sign']
                    #Deal with bifurcation events and overshoots in every edge
                    #bifRBCsIndes - array with overshooting RBCs from smallest to largest index
                    bifRBCsIndex=[]
                    nRBC=len(e['rRBC'])
                    if sign == 1.0:
                        if e['rRBC'][-1] > e['length']:
                            for i,j in enumerate(e['rRBC'][::-1]):
                                if j > e['length']:
                                    bifRBCsIndex.append(nRBC-1-i)
                                else:
                                    break
                        bifRBCsIndex=bifRBCsIndex[::-1]
                    else:
                        if e['rRBC'][0] < 0:
                            for i,j in enumerate(e['rRBC']):
                                if j < 0:
                                    bifRBCsIndex.append(i)
                                else:
                                    break
                    noBifEvents=len(bifRBCsIndex)
                else:
                    noBifEvents = 0
                #Convergent Edge without a bifurcation event
                if noBifEvents == 0 and G.vs[vi]['vType']==4:
                    convEdges2.append(ei)
                if noBifEvents == 0:
                    e['nRBCint']=len(e['rRBC'])
        #-------------------------------------------------------------------------------------------
                #Check if a bifurcation event is taking place
                if noBifEvents > 0:
                    #If Edge is outlflow Edge --> remove RBCs
                    if G.vs[vi]['vType'] == 2:
                        overshootsNo=noBifEvents
                        nRBCint=e['nRBCint']
                        if e['sign'] == 1:
                            for i in range(noBifEvents):
                                nRBCint += self._nRBC_mean_outflow(rRBCold[-1*(i+1)],e,dt)
                        else:
                            for i in range(noBifEvents):
                                nRBCint += self._nRBC_mean_outflow(rRBCold[i],e,dt)
                        e['rRBC']=[e['rRBC'][:-noBifEvents] if sign == 1.0 else e['rRBC'][noBifEvents::]][0]
                        nRBCint += len(e['rRBC'])
                        e['nRBCint']=nRBCint
                        vertexUpdate.append(e.target)
                        vertexUpdate.append(e.source)
                        edgeUpdate.append(ei)
            #-------------------------------------------------------------------------------------------
                    #if vertex is connecting vertex
                    elif G.vs[vi]['vType'] == 5:
                        outE=G.vs[vi]['outflowE'][0]
                        oe=G.es[outE]
                        #Calculate possible number of bifurcation Events
                        #distToFirst = distance to first vertex in vessel
                        if len(oe['rRBC']) > 0:
                            distToFirst=oe['rRBC'][0] if oe['sign'] == 1.0 else oe['length']-oe['rRBC'][-1]
                        else:
                            distToFirst=oe['length']
                        #Check how many RBCs fit into the new Vessel
                        posNoBifEvents=int(np.floor(distToFirst/oe['minDist']))
                        #Check how many RBCs are allowed by nMax --> limitation results from np.floor(length/minDist) 
                        #and that RBCs are only 'half' in the vessel 
                        if posNoBifEvents + len(oe['rRBC']) > oe['nMax']:
                            posNoBifEvents = int(oe['nMax'] - len(oe['rRBC']))
                        #OvershootsNo: compare posNoBifEvents with noBifEvents
                        #posBifRBCsIndex --> array with possible number of bifurcations taking place
                        if posNoBifEvents > noBifEvents:
                            posBifRBCsIndex = bifRBCsIndex
                            overshootsNo=noBifEvents
                        elif posNoBifEvents == 0:
                            posBifRBCsIndex=[]
                            overshootsNo=0
                        else:
                            posBifRBCsIndex=[bifRBCsIndex[-posNoBifEvents::] if sign == 1.0 \
                                else bifRBCsIndex[:posNoBifEvents]]
                            overshootsNo=posNoBifEvents
                        if overshootsNo > 0:
                            #overshootsDist --> array with the distances which the RBCs overshoot, 
                            #starts wiht the RBC which overshoots the least 
                            overshootDist=[e['rRBC'][posBifRBCsIndex]-[e['length']]*overshootsNo if sign == 1.0
                                else [0]*overshootsNo-e['rRBC'][posBifRBCsIndex]][0]
                            if sign != 1.0:
                                overshootDist = overshootDist[::-1]
                            #overshootTime --> time which every RBCs overshoots
                            overshootTime=overshootDist / ([e['v']]*overshootsNo)
                            #position --> where the overshooting RBCs would be located in the outEdge
                            position=np.array(overshootTime)*np.array([oe['v']]*overshootsNo)
                            #Check if RBCs overshoot the whole downstream vessel
                            if len(oe['rRBC']) == 0:
                                if position[-1] > oe['length']:
                                    position = np.array(position)-np.array([position[-1]-oe['length']]*len(position))
                            #Maxmimum number of overshoots possible is infact limited by the overshootDistance of the first RBC
                            #If the RBCs travel with the same speed than the bulk flow this check is not necessary
                            #BUT due to different velocity factors RBCs cann "ran into each other" at connecting bifurcations
                            overshootsNoReduce=0
                            #Check if RBCs ran into another
                            for i in range(overshootsNo-1):
                                index=-1*(i+1)
                                if position[index]-position[index-1] < oe['minDist']:
                                    position[index-1]=position[index]-oe['minDist']
                                if position[index-1] < 0:
                                    overshootsNoReduce += 1
                            overshootsNo = overshootsNo-overshootsNoReduce
                            position=position[-1*overshootsNo::]
                            #Check if the RBCs overshoots RBCs present in the outflow vessel
                            overshootsNoReduce2=0
                            if len(oe['rRBC']) > 0:
                                if oe['sign'] == 1 and position[-1] > oe['rRBC'][0]-oe['minDist']:
                                    posLead=position[-1]
                                    position = np.array(position)-np.array([posLead-(oe['rRBC'][0]-oe['minDist'])]*len(position))
                                    for i in range(overshootsNo):
                                        if position[i] < 0:
                                            overshootsNoReduce2 += 1
                                        else:
                                            break
                                elif oe['sign'] == -1 and position[-1] > oe['length']-oe['rRBC'][-1]-oe['minDist']:
                                    posLead=position[-1]
                                    position = np.array(position)-np.array([posLead-(oe['length']-oe['rRBC'][-1]-oe['minDist'])]*len(position))
                                    for i in range(overshootsNo):
                                        if position[i] < 0:
                                            overshootsNoReduce2 += 1
                                        else:
                                            break
                            overshootsNo = overshootsNo-overshootsNoReduce2
                            if overshootsNo == 0:
                                position = []
                            else:
                                position=position[-1*overshootsNo::]
                            #Add rbcs to new Edge
                            if overshootsNo > 0:
                                if oe['sign'] == 1.0:
                                    oe['rRBC']=np.concatenate([position, oe['rRBC']])
                                else:
                                    position = [oe['length']]*overshootsNo - position[::-1]
                                    oe['rRBC']=np.concatenate([oe['rRBC'],position])
                                nRBCint=oe['nRBCint']
                                for i in position:
                                    nRBCint += self._nRBC_mean_inflow(i,oe,dt)
                                oe['nRBCint']=nRBCint
                            #Remove RBCs from old Edge
                                nRBCint=e['nRBCint']
                                if e['sign'] == 1:
                                    for i in range(overshootsNo):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold[-1*(i+1)],e,dt)
                                else:
                                    for i in range(overshootsNo):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold[i],e,dt)
                                if sign == 1.0:
                                    e['rRBC']=e['rRBC'][:-overshootsNo]
                                else:
                                    e['rRBC']=e['rRBC'][overshootsNo::]
                                nRBCint += len(e['rRBC'])
                                e['nRBCint']=nRBCint
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        noStuckRBCs=len(bifRBCsIndex)-overshootsNo
                        #move stuck RBCs back into vessel
                        for i in range(noStuckRBCs):
                            index=-1*(i+1) if sign == 1.0 else i
                            e['rRBC'][index]=e['length']-i*e['minDist'] if sign == 1.0 else 0+i*e['minDist']
                        #Recheck if the distance between the newly introduces RBCs is still big enough 
                        if len(e['rRBC']) > 1:
                            moved = 0
                            count = 0
                            if sign == 1.0:
                                for i in range(-1,-1*(len(e['rRBC'])),-1):
                                    index=i-1
                                    if e['rRBC'][i] < e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]-e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs and moved == 0:
                                        break
                            else:
                                for i in range(len(e['rRBC'])-1):
                                    index=i+1
                                    if e['rRBC'][i] > e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]+e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs+1 and moved == 0:
                                        break
          #-------------------------------------------------------------------------------------------
                    #if vertex is divergent vertex
                    elif G.vs[vi]['vType'] == 3:
                        outEdges=G.vs[vi]['outflowE']
                        outE=outEdges[0]
                        outE2=outEdges[1]
                        #Check if there are two or three outEdgs
                        if len(outEdges) > 2:
                            outE3=outEdges[2]
                        #Differ between capillaries and non-capillaries
                        if G.vs[vi]['isCap']:
                            nonCap = 0
                            preferenceList = [x[1] for x in sorted(zip(np.array(G.es[outEdges]['flow'])/np.array(G.es[outEdges]['crosssection']), \
                                outEdges), reverse=True)]
                        else:
                            nonCap = 1
                            preferenceList = [x[1] for x in sorted(zip(G.es[outEdges]['flow'], outEdges), reverse=True)]
                            #Check if the divergent bifurcation has degree 4
                            if len(outEdges) > 2:
                                ratio1 = G.es[preferenceList[0]]['flow']/e['flow']
                                ratio2 = G.es[preferenceList[1]]['flow']/e['flow']
                                ratio3 = G.es[preferenceList[2]]['flow']/e['flow']
                            else:
                                ratio1 = Physiol.phase_separation_effect(G.es[preferenceList[0]]['flow']/e['flow'], \
                                    G.es[preferenceList[0]]['diameter'],G.es[preferenceList[1]]['diameter'],e['diameter'],e['htd'])
                                ratio2 = 1.0 -  ratio1
                                ratio3 = 0
                        #Define prefered OutEdges based on bifurcation rule
                        outEPref=preferenceList[0]
                        outEPref2=preferenceList[1]
                        oe=G.es[outEPref]
                        oe2=G.es[outEPref2]
                        if len(outEdges) > 2:
                            outEPref3 = preferenceList[2]
                            oe3=G.es[outEPref3]
                        #Calculate distance to first RBC for outEPref
                        if len(oe['rRBC']) > 0:
                            distToFirst=oe['rRBC'][0] if oe['sign'] == 1.0 \
                                else oe['length']-oe['rRBC'][-1]
                        else:
                            distToFirst=oe['length']
                        #Calculate distance to first RBC for outEPref2
                        if len(oe2['rRBC']) > 0:
                            distToFirst2=oe2['rRBC'][0] if oe2['sign'] == 1.0 \
                                else oe2['length']-oe2['rRBC'][-1]
                        else:
                            distToFirst2=oe2['length']
                        #Calculate distance to first RBC for outEPref3 (if it exists)
                        if len(outEdges) > 2:
                            if len(oe3['rRBC']) > 0:
                                distToFirst3=oe3['rRBC'][0] if oe3['sign'] == 1.0 \
                                    else oe3['length']-oe3['rRBC'][-1]
                            else:
                                distToFirst3=oe3['length']
                        #Check how many RBCs are allowed by nMax for outEPref
                        posNoBifEventsPref=np.floor(distToFirst/oe['minDist'])
                        if posNoBifEventsPref + len(oe['rRBC']) > oe['nMax']:
                            posNoBifEventsPref = oe['nMax'] - len(oe['rRBC'])
                        #Check how many RBCs are allowed by nMax for outEPref2
                        posNoBifEventsPref2=np.floor(distToFirst2/oe2['minDist'])
                        if posNoBifEventsPref2 + len(oe2['rRBC']) > oe2['nMax']:
                            posNoBifEventsPref2 = oe2['nMax'] - len(oe2['rRBC'])
                        #Check how many RBCs are allowed by nMax for outEPref3
                        if len(outEdges) > 2:
                            posNoBifEventsPref3=np.floor(distToFirst3/oe3['minDist'])
                            if posNoBifEventsPref3 + len(oe3['rRBC']) > oe3['nMax']:
                                posNoBifEventsPref3 = oe3['nMax'] - len(oe3['rRBC'])
                        else:
                            posNoBifEventsPref3 = 0
                        #Calculate total number of bifurcation events possible
                        posNoBifEvents=int(posNoBifEventsPref+posNoBifEventsPref2+posNoBifEventsPref3)
                        #Compare possible number of bifurcation events with number of bifurcations taking place
                        if posNoBifEvents > noBifEvents:
                            posBifRBCsIndex=bifRBCsIndex
                            overshootsNo=noBifEvents
                        elif posNoBifEvents == 0:
                            posBifRBCsIndex=[]
                            overshootsNo=0
                        else:
                            posBifRBCsIndex=[bifRBCsIndex[-posNoBifEvents::] if sign == 1.0 \
                                else bifRBCsIndex[:posNoBifEvents]]
                            overshootsNo=posNoBifEvents
                        if nonCap:
                            overshootsNo1 = np.floor(ratio1 * overshootsNo)
                            overshootsNo3 = np.floor(ratio3 * overshootsNo)
                            overshootsNo2 = overshootsNo - overshootsNo1 - overshootsNo3
                            if overshootsNo1 > posNoBifEventsPref:
                                overshootsNo1 = posNoBifEventsPref
                            if overshootsNo2 > posNoBifEventsPref2:
                                overshootsNo2 = posNoBifEventsPref2
                            if overshootsNo3 > posNoBifEventsPref3:
                                overshootsNo3 = posNoBifEventsPref3
                            overshootsNo = int(overshootsNo1 + overshootsNo2 + overshootsNo3)
                            posNoBifEvents = overshootsNo
                            posBifRBCsIndex=[bifRBCsIndex[-posNoBifEvents::] if sign == 1.0 \
                                else bifRBCsIndex[:posNoBifEvents]]
                        if overshootsNo > 0:
                            #overshootDist starts with the RBC which overshoots the least
                            overshootDist=[e['rRBC'][posBifRBCsIndex]-[e['length']]*overshootsNo if sign == 1.0
                                else [0]*overshootsNo-e['rRBC'][posBifRBCsIndex]][0]
                            if sign != 1.0:
                                overshootDist = overshootDist[::-1]
                            #overshootTime starts with the RBC which overshoots the least
                            overshootTime=overshootDist / ([e['v']]*overshootsNo)
                            #Calculate position of overshootRBCs in every outEdge
                            #the values in position are stored such that they can directly concatenated with outE['rRBC']
                            #flow direction of outEdge is considered
                            #position = [pos_min ... pos_max]
                            if oe['sign'] == 1.0:
                                position1=np.array(overshootTime)*np.array([oe['v']]*overshootsNo)
                            else:
                                position1=np.array([oe['length']]*overshootsNo)-np.array(overshootTime[::-1])* \
                                    np.array([oe['v']]*overshootsNo)
                            if oe2['sign'] == 1.0:
                                position2=np.array(overshootTime)*np.array([oe2['v']]*overshootsNo)
                            else:
                                position2=np.array([oe2['length']]*overshootsNo)-np.array(overshootTime[::-1])* \
                                    np.array([oe2['v']]*overshootsNo)
                            if len(outEdges) > 2:
                                if oe3['sign'] == 1.0:
                                    position3=np.array(overshootTime)*np.array([oe3['v']]*overshootsNo)
                                else:
                                    position3=np.array([oe3['length']]*overshootsNo)-np.array(overshootTime[::-1])* \
                                        np.array([oe3['v']]*overshootsNo)
                            if nonCap:
                                countNo1=0
                                countNo2=0
                                countNo3=0
                                positionPref3=[]
                                positionPref2=[]
                                positionPref1=[]
                                last=3
                                counterPref3=[]
                                counterPref2=[]
                                counterPref1=[]
                                for i in range(overshootsNo):
                                    index=-1*(i+1) if sign == 1.0 else i
                                    index1=-1*(i+1) if oe['sign'] == 1.0 else i
                                    index2=-1*(i+1) if oe2['sign'] == 1.0 else i
                                    if len(outEdges) > 2:
                                        index3=-1*(i+1) if oe3['sign'] == 1.0 else i
                                    if last == 3:
                                        if countNo1 < overshootsNo1:
                                            if positionPref1 == []:
                                                if len(oe['rRBC']) > 0:
                                                    if oe['sign'] == 1:
                                                        if position1[index1] > oe['rRBC'][0]-oe['minDist']:
                                                            positionPref1.append(oe['rRBC'][0]-oe['minDist'])
                                                        else:
                                                            positionPref1.append(position1[index1])
                                                    else:
                                                        if position1[index1] < oe['rRBC'][-1]+oe['minDist']:
                                                            positionPref1.append(oe['rRBC'][-1]+oe['minDist'])
                                                        else:
                                                            positionPref1.append(position1[index1])
                                                else:
                                                    if oe['sign'] == 1:
                                                        if position1[index1] > oe['length']:
                                                            positionPref1.append(oe['length'])
                                                        else:
                                                            positionPref1.append(position1[index1])
                                                    else:
                                                        if position1[index1] < 0:
                                                            positionPref1.append(0)
                                                        else:
                                                            positionPref1.append(position1[index1])
                                            else:
                                                positionPref1.append(position1[index1])
                                            counterPref1.append(index)
                                            countNo1 += 1
                                            last = 1
                                        else: 
                                            if countNo2 < overshootsNo2:
                                                if positionPref2 == []:
                                                    if len(oe2['rRBC']) > 0:
                                                        if oe2['sign'] == 1:
                                                            if position2[index2] > oe2['rRBC'][0]-oe2['minDist']:
                                                                positionPref2.append(oe2['rRBC'][0]-oe2['minDist'])
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                        else:
                                                            if position2[index2] < oe2['rRBC'][-1]+oe2['minDist']:
                                                                positionPref2.append(oe2['rRBC'][-1]+oe2['minDist'])
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                    else:
                                                        if oe2['sign'] == 1:
                                                            if position2[index2] > oe2['length']:
                                                                positionPref2.append(oe2['length'])
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                        else:
                                                            if position2[index2] < 0:
                                                                positionPref2.append(0)
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                else:
                                                    positionPref2.append(position2[index2])
                                                counterPref2.append(index)
                                                countNo2 += 1
                                                last = 2
                                            elif countNo3 < overshootsNo3:
                                                if positionPref3 == []:
                                                    if len(oe3['rRBC']) > 0:
                                                        if oe3['sign'] == 1:
                                                            if position3[index3] > oe3['rRBC'][0]-oe3['minDist']:
                                                                positionPref3.append(oe3['rRBC'][0]-oe3['minDist'])
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                        else:
                                                            if position3[index3] < oe3['rRBC'][-1]+oe3['minDist']:
                                                                positionPref3.append(oe3['rRBC'][-1]+oe3['minDist'])
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                    else:
                                                        if oe3['sign'] == 1:
                                                            if position3[index3] > oe3['length']:
                                                                positionPref3.append(oe3['length'])
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                        else:
                                                            if position3[index3] < 0:
                                                                positionPref3.append(0)
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                else:
                                                    positionPref3.append(position3[index3])
                                                counterPref3.append(index)
                                                countNo3 += 1
                                                last = 2
                                            else:
                                                print('BIGERROR all overshootRBCS should fit')
                                    elif last == 1:
                                        if countNo2 < overshootsNo2:
                                            if positionPref2 == []:
                                                if len(oe2['rRBC']) > 0:
                                                    if oe2['sign'] == 1:
                                                        if position2[index2] > oe2['rRBC'][0]-oe2['minDist']:
                                                            positionPref2.append(oe2['rRBC'][0]-oe2['minDist'])
                                                        else:
                                                            positionPref2.append(position2[index2])
                                                    else:
                                                        if position2[index2] < oe2['rRBC'][-1]+oe2['minDist']:
                                                            positionPref2.append(oe2['rRBC'][-1]+oe2['minDist'])
                                                        else:
                                                            positionPref2.append(position2[index2])
                                                else:
                                                    if oe2['sign'] == 1:
                                                        if position2[index2] > oe2['length']:
                                                            positionPref2.append(oe2['length'])
                                                        else:
                                                            positionPref2.append(position2[index2])
                                                    else:
                                                        if position2[index2] < 0:
                                                            positionPref2.append(0)
                                                        else:
                                                            positionPref2.append(position2[index2])
                                            else:
                                                positionPref2.append(position2[index2])
                                            counterPref2.append(index)
                                            countNo2 += 1
                                            last = 2
                                        else: 
                                            if countNo3 < overshootsNo3:
                                                if positionPref3 == []:
                                                    if len(oe3['rRBC']) > 0:
                                                        if oe3['sign'] == 1:
                                                            if position3[index3] > oe3['rRBC'][0]-oe3['minDist']:
                                                                positionPref3.append(oe3['rRBC'][0]-oe3['minDist'])
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                        else:
                                                            if position3[index3] < oe3['rRBC'][-1]+oe3['minDist']:
                                                                positionPref3.append(oe3['rRBC'][-1]+oe3['minDist'])
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                    else:
                                                        if oe3['sign'] == 1:
                                                            if position3[index3] > oe3['length']:
                                                                positionPref3.append(oe3['length'])
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                        else:
                                                            if position3[index3] < 0:
                                                                positionPref3.append(0)
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                else:
                                                    positionPref3.append(position3[index3])
                                                counterPref3.append(index)
                                                countNo3 += 1
                                                last = 3
                                            elif countNo1 < overshootsNo1:
                                                if positionPref1 == []:
                                                    if len(oe['rRBC']) > 0:
                                                        if oe['sign'] == 1:
                                                            if position1[index1] > oe['rRBC'][0]-oe['minDist']:
                                                                positionPref1.append(oe['rRBC'][0]-oe['minDist'])
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                        else:
                                                            if position1[index1] < oe['rRBC'][-1]+oe['minDist']:
                                                                positionPref1.append(oe['rRBC'][-1]+oe['minDist'])
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                    else:
                                                        if oe['sign'] == 1:
                                                            if position1[index1] > oe['length']:
                                                                positionPref1.append(oe['length'])
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                        else:
                                                            if position1[index1] < 0:
                                                                positionPref1.append(0)
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                else:
                                                    positionPref1.append(position1[index1])
                                                counterPref1.append(index)
                                                countNo1 += 1
                                                last = 1
                                            else:
                                                print('BIGERROR all overshootRBCS should fit')
                                    elif last == 2:
                                        if countNo3 < overshootsNo3:
                                            if positionPref3 == []:
                                                if len(oe3['rRBC']) > 0:
                                                    if oe3['sign'] == 1:
                                                        if position3[index3] > oe3['rRBC'][0]-oe3['minDist']:
                                                            positionPref3.append(oe3['rRBC'][0]-oe3['minDist'])
                                                        else:
                                                            positionPref3.append(position3[index3])
                                                    else:
                                                        if position3[index3] < oe3['rRBC'][-1]+oe3['minDist']:
                                                            positionPref3.append(oe3['rRBC'][-1]+oe3['minDist'])
                                                        else:
                                                            positionPref3.append(position3[index3])
                                                else:
                                                    if oe3['sign'] == 1:
                                                        if position3[index3] > oe3['length']:
                                                            positionPref3.append(oe3['length'])
                                                        else:
                                                            positionPref3.append(position3[index3])
                                                    else:
                                                        if position3[index3] < 0:
                                                            positionPref3.append(0)
                                                        else:
                                                            positionPref3.append(position3[index3])
                                            else:
                                                positionPref3.append(position3[index3])
                                            counterPref3.append(index)
                                            countNo3 += 1
                                            last = 3
                                        else: 
                                            if countNo1 < overshootsNo1:
                                                if positionPref1 == []:
                                                    if len(oe['rRBC']) > 0:
                                                        if oe['sign'] == 1:
                                                            if position1[index1] > oe['rRBC'][0]-oe['minDist']:
                                                                positionPref1.append(oe['rRBC'][0]-oe['minDist'])
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                        else:
                                                            if position1[index1] < oe['rRBC'][-1]+oe['minDist']:
                                                                positionPref1.append(oe['rRBC'][-1]+oe['minDist'])
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                    else:
                                                        if oe['sign'] == 1:
                                                            if position1[index1] > oe['length']:
                                                                positionPref1.append(oe['length'])
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                        else:
                                                            if position1[index1] < 0:
                                                                positionPref1.append(0)
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                else:
                                                    positionPref1.append(position1[index1])
                                                counterPref1.append(index)
                                                countNo1 += 1
                                                last = 1
                                            elif countNo2 < overshootsNo2:
                                                if positionPref2 == []:
                                                    if len(oe2['rRBC']) > 0:
                                                        if oe2['sign'] == 1:
                                                            if position2[index2] > oe2['rRBC'][0]-oe2['minDist']:
                                                                positionPref2.append(oe2['rRBC'][0]-oe2['minDist'])
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                        else:
                                                            if position2[index2] < oe2['rRBC'][-1]+oe2['minDist']:
                                                                positionPref2.append(oe2['rRBC'][-1]+oe2['minDist'])
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                    else:
                                                        if oe2['sign'] == 1:
                                                            if position2[index2] > oe2['length']:
                                                                positionPref2.append(oe2['length'])
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                        else:
                                                            if position2[index2] < 0:
                                                                positionPref2.append(0)
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                else:
                                                    positionPref2.append(position2[index2])
                                                counterPref2.append(index)
                                                countNo2 += 1
                                                last = 2
                                            else:
                                                print('BIGERROR all overshootRBCS should fit')
                                    # make sure that distance between adjacent RBCs is large enough
                                    if last == 1:
                                        if len(positionPref1) >= 2:
                                            if oe['sign'] == 1:
                                                if positionPref1[-1] > positionPref1[-2] or positionPref1[-2]-positionPref1[-1] < oe['minDist']-eps:
                                                    positionPref1[-1] = positionPref1[-2] - oe['minDist']
                                            else:
                                                if positionPref1[-1] < positionPref1[-2] or positionPref1[-1]-positionPref1[-2] < oe['minDist']-eps:
                                                    positionPref1[-1] = positionPref1[-2] + oe['minDist']
                                    elif last == 2:
                                        if len(positionPref2) >= 2:
                                            if oe2['sign'] == 1:
                                                if positionPref2[-1] > positionPref2[-2] or positionPref2[-2] - positionPref2[-1] < oe2['minDist']-eps:
                                                    positionPref2[-1] = positionPref2[-2] - oe2['minDist']
                                            else:
                                                if positionPref2[-1] < positionPref2[-2] or positionPref2[-1] - positionPref2[-2] < oe2['minDist']-eps:
                                                    positionPref2[-1] = positionPref2[-2] + oe2['minDist']
                                    elif last == 3:
                                        if len(positionPref3) >= 2:
                                            if oe3['sign'] == 1:
                                                if positionPref3[-1] > positionPref3[-2] or positionPref3[-2] - positionPref3[-1] < oe3['minDist']-eps:
                                                    positionPref3[-1] = positionPref3[-2] - oe3['minDist']
                                            else:
                                                if positionPref3[-1] < positionPref3[-2] or positionPref3[-1] - positionPref3[-2] < oe3['minDist']-eps:
                                                    positionPref3[-1] = positionPref3[-2] + oe3['minDist']
                                if positionPref1 != []:
                                    if oe['sign'] == 1:
                                        if positionPref1[-1] < 0:
                                            positionPref1[-1] = 0
                                            for i in range(-1,-1*(len(positionPref1)),-1):
                                                if positionPref1[i-1]-positionPref1[i] < oe['minDist'] - eps:
                                                    positionPref1[i-1]=positionPref1[i] + oe['minDist']
                                                else:
                                                    break
                                            if len(oe['rRBC']) > 0:
                                                if positionPref1[0] > oe['rRBC'][0]:
                                                    print('BIGERROR should not happen 5')
                                                    print(positionPref1[0])
                                                    print(oe['rRBC'][0])
                                                    print(len(positionPref1))
                                                    print(posOld)
                                        if positionPref1[0] > oe['length']:
                                            positionPref1[0] = oe['length']
                                            for i in range(len(positionPref1)-1):
                                                if positionPref1[i]-positionPref1[i+1] < oe['minDist'] - eps:
                                                    positionPref1[i+1]=positionPref1[i] - oe['minDist']
                                                else:
                                                    break
                                            if positionPref1[-1] < 0:
                                                print('BIGERROR should not happen 5_2')
                                                print(positionPref1[-1])
                                    else:
                                        if positionPref1[-1] > oe['length']:
                                            positionPref1[-1] = oe['length']
                                            for i in range(-1,-1*(len(positionPref1)),-1):
                                                if positionPref1[i]-positionPref1[i-1] < oe['minDist'] - eps:
                                                    positionPref1[i-1]=positionPref1[i] - oe['minDist']
                                                else:
                                                    break
                                            if len(oe['rRBC']) > 0:
                                                if positionPref1[0] < oe['rRBC'][-1]:
                                                    print('BIGERROR should not happen 6')
                                                    print(positionPref1[0])
                                                    print(oe['rRBC'][-1])
                                                    print(len(positionPref1))
                                                    print(posOld)
                                        if positionPref1[0] < 0:
                                            positionPref1[0] = 0
                                            for i in range(len(positionPref1)-1):
                                                if positionPref1[i+1]-positionPref1[i] < oe['minDist'] - eps:
                                                    positionPref1[i+1]=positionPref1[i] + oe['minDist']
                                                else:
                                                    break
                                            if positionPref1[-1] > oe['length']:
                                                print('BIGERROR should not happen 6_2')
                                                print(positionPref1[-1])
                                                print(oe['length'])
                                if positionPref2 != []:
                                    if oe2['sign'] == 1:
                                        if positionPref2[-1] < 0:
                                            positionPref2[-1] = 0
                                            for i in range(-1,-1*(len(positionPref2)),-1):
                                                if positionPref2[i-1]-positionPref2[i] < oe2['minDist'] + eps:
                                                    positionPref2[i-1]=positionPref2[i] + oe2['minDist']
                                                else:
                                                    break
                                            if len(oe2['rRBC']) > 0:
                                                if positionPref2[0] > oe2['rRBC'][0]:
                                                    print('BIGERROR should not happen 7')
                                                    print(positionPref2[0])
                                                    print(oe2['rRBC'][0])
                                                    print(len(positionPref2))
                                                    print(posOld)
                                        if positionPref2[0] > oe2['length']:
                                            positionPref2[0] = oe2['length']
                                            for i in range(len(positionPref2)-1):
                                                if positionPref2[i]-positionPref2[i+1] < oe2['minDist'] + eps:
                                                    positionPref2[i+1]=positionPref2[i] - oe2['minDist']
                                                else:
                                                    break
                                            if positionPref2[-1] < 0:
                                                print('BIGERROR should not happen 7_2')
                                                print(positionPref2[-1])
                                    else:
                                        if positionPref2[-1] > oe2['length']:
                                            positionPref2[-1] = oe2['length']
                                            for i in range(-1,-1*(len(positionPref2)),-1):
                                                if positionPref2[i]-positionPref2[i-1] < oe2['minDist'] + eps:
                                                    positionPref2[i-1]=positionPref2[i] - oe2['minDist']
                                                else:
                                                    break
                                            if len(oe2['rRBC']) > 0:
                                                if positionPref2[0] < oe2['rRBC'][-1]:
                                                    print('BIGERROR should not happen 8')
                                                    print(positionPref2[0])
                                                    print(oe2['rRBC'][-1])
                                                    print(len(positionPref2))
                                                    print(posOld)
                                        if positionPref2[0] < 0:
                                            positionPref2[0] = 0
                                            for i in range(len(positionPref2)-1):
                                                if positionPref2[i+1]-positionPref2[i] < oe2['minDist'] + eps:
                                                    positionPref2[i+1]=positionPref2[i] + oe2['minDist']
                                                else:
                                                    break
                                            if positionPref2[-1] > oe2['length']:
                                                print('BIGERROR should not happen 8_2')
                                                print(positionPref2[-1])
                                                print(oe2['length'])
                                if positionPref3 != []:
                                    if oe3['sign'] == 1:
                                        if positionPref3[-1] < 0:
                                            positionPref3[-1] = 0
                                            for i in range(-1,-1*(len(positionPref3)),-1):
                                                if positionPref3[i-1]-positionPref3[i] < oe3['minDist'] + eps:
                                                    positionPref3[i-1]=positionPref3[i] + oe3['minDist']
                                                else:
                                                    break
                                            if len(oe3['rRBC']) > 0:
                                                if positionPref3[0] > oe3['rRBC'][0]:
                                                    print('BIGERROR should not happen 9')
                                                    print(positionPref3[0])
                                                    print(oe3['rRBC'][0])
                                                    print(len(positionPref3))
                                                    print(posOld)
                                        if positionPref3[0] > oe3['length']:
                                            positionPref3[0] = oe3['length']
                                            for i in range(len(positionPref3)-1):
                                                if positionPref3[i]-positionPref3[i+1] < oe3['minDist'] + eps:
                                                    positionPref2[i+1]=positionPref3[i] - oe3['minDist']
                                                else:
                                                    break
                                            if positionPref3[-1] < 0:
                                                print('BIGERROR should not happen 9_2')
                                                print(positionPref3[-1])
                                    else:
                                        if positionPref3[-1] > oe3['length']:
                                            positionPref3[-1] = oe3['length']
                                            for i in range(-1,-1*(len(positionPref3)),-1):
                                                if positionPref3[i]-positionPref3[i-1] < oe3['minDist'] + eps:
                                                    positionPref3[i-1]=positionPref3[i] - oe3['minDist']
                                                else:
                                                   break
                                            if len(oe3['rRBC']) > 0:
                                                if positionPref3[0] < oe3['rRBC'][-1]:
                                                    print('BIGERROR should not happen 10')
                                                    print(positionPref3[0])
                                                    print(oe3['rRBC'][-1])
                                                    print(len(positionPref3))
                                                    print(posOld)
                                        if positionPref3[0] < 0:
                                            positionPref3[0] = 0
                                            for i in range(len(positionPref3)-1):
                                                if positionPref3[i+1]-positionPref3[i] < oe3['minDist'] + eps:
                                                    positionPref3[i+1]=positionPref3[i] + oe3['minDist']
                                                else:
                                                    break
                                            if positionPref3[-1] > oe3['length']:
                                                print('BIGERROR should not happen 10_2')
                                                print(positionPref3[-1])
                                                print(oe3['length'])
                            else:
                                #To begin with it is tried if all RBCs fit into the prefered outEdge. The time of arrival at the RBCs is take into account
                                #RBCs which would be too close together are put into the other edge
                                #postion2/position3 is used if there is not enough space in the prefered outEdge and hence the RBC is moved to the other outEdge
                                positionPref3=[]
                                positionPref2=[]
                                positionPref1=[]
                                counterPref3=[]
                                counterPref2=[]
                                counterPref1=[]
                                #number of RBCs in the Edges
                                countPref1=0
                                countPref2=0
                                countPref3=0
                                pref1Full=0
                                pref2Full=0
                                pref3Full=0
                                #Loop over all movable RBCs (begin with RBC which overshot the most)
                                for i in range(overshootsNo):
                                    index=-1*(i+1) if sign == 1.0 else i
                                    index1=-1*(i+1) if oe['sign'] == 1.0 else i
                                    index2=-1*(i+1) if oe2['sign'] == 1.0 else i
                                    #The possible number of RBCs results from the distance the first RBC overshoots
                                    #it can happen that due to that more RBCs are blocked than expected, that is checked with the following values
                                    if len(outEdges) > 2:
                                        index3=-1*(i+1) if oe3['sign'] == 1.0 else i
                                    #check if RBC still fits into Prefered OutE
                                    if posNoBifEventsPref > countPref1 and pref1Full == 0:
                                        #Check if there are RBCs present in outEPref
                                        #RBCs have already been put into outEPref1
                                        if positionPref1 != []: 
                                            #Check if distance to preceding RBC is big enough
                                            dist1=positionPref1[-1]-position1[index1] if oe['sign'] == 1.0 \
                                                else position1[index1]-positionPref1[-1]
                                            #If the distance is not big enough check if RBC fits into outEPref2
                                            if dist1 < oe['minDist']:
                                                #if RBCs are present in the outEdgePref2
                                                #check if RBC still fits into outEPref2
                                                if posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                                    if positionPref2 != []:
                                                        #Check if distance to preceding RBC is big enough
                                                        dist2=positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                                            else position2[index2]-positionPref2[-1]
                                                        #If the distance is not big enough check if RBC fits into outEPref3
                                                        if dist2 < oe2['minDist']:
                                                            #Check if there is a third outEdge
                                                            if len(outEdges)>2:
                                                                #check if RBC still fits into outEPref3
                                                                if posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                                                    #Check if there are RBCs in the third outEdge
                                                                    if positionPref3 != []: 
                                                                        #Check if distance to preceding RBC is big enough
                                                                        dist3=positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                                            else position3[index3]-positionPref3[-1]
                                                                        #if there is not enough space in the third outEdge
                                                                        #Check in which edge the RBC is blocked the shortest time
                                                                        if dist3 < oe3['minDist']:
                                                                            space1 =  positionPref1[-1] if oe['sign'] == 1.0 \
                                                                                else oe['length']-positionPref1[-1]
                                                                            if np.floor(space1/oe['minDist']) >= 1:
                                                                                timeBlocked1=(oe['minDist']-dist1)/oe['v']
                                                                            else:
                                                                                timeBlocked1=None
                                                                                pref1Full=1
                                                                            space2 =  positionPref2[-1] if oe2['sign'] == 1.0 \
                                                                                else oe2['length']-positionPref2[-1]
                                                                            if np.floor(space2/oe2['minDist']) >= 1:
                                                                                timeBlocked2=(oe2['minDist']-dist2)/oe2['v']
                                                                            else:
                                                                                timeBlocked2=None
                                                                                pref2Full=1
                                                                            space3 =  positionPref3[-1] if oe3['sign'] == 1.0 \
                                                                                else oe3['length']-positionPref3[-1]
                                                                            if np.floor(space3/oe3['minDist']) >= 1:
                                                                                timeBlocked3=(oe3['minDist']-dist3)/oe3['v']
                                                                            else:
                                                                                timeBlocked3=None
                                                                                pref3Full=1
                                                                            if pref1Full == 1 and pref2Full == 1 and pref3Full == 1:
                                                                                break
                                                                            #Define newOutEdge
                                                                            newOutEdge=0
                                                                            if timeBlocked1 == None: #2 or 3
                                                                                if timeBlocked2 == None:
                                                                                    newOutEdge=3
                                                                                elif timeBlocked3 == None:
                                                                                    newOutEdge=2
                                                                                elif timeBlocked2 <= timeBlocked3:
                                                                                    newOutEdge=2
                                                                                elif timeBlocked3 <= timeBlocked2:
                                                                                    newOutEdge=3
                                                                            elif timeBlocked2 == None: #1 or 3
                                                                                if timeBlocked3 == None:
                                                                                    newOutEdge=1
                                                                                elif timeBlocked1 <= timeBlocked3:
                                                                                    newOutEdge=1
                                                                                elif timeBlocked3 <= timeBlocked1:
                                                                                    newOutEdge=3
                                                                            elif timeBlocked3 == None: #1 or 2
                                                                                if timeBlocked2 <= timeBlocked1:
                                                                                    newOutEdge=2
                                                                                elif timeBlocked1 <= timeBlocked2:
                                                                                    newOutEdge=1
                                                                            else:
                                                                                if np.min([timeBlocked1,timeBlocked2,timeBlocked3]) == timeBlocked1:
                                                                                    newOutEdge=1
                                                                                elif np.min([timeBlocked1,timeBlocked2,timeBlocked3]) == timeBlocked2:
                                                                                    newOutEdge=2
                                                                                elif np.min([timeBlocked1,timeBlocked2,timeBlocked3]) == timeBlocked3:
                                                                                    newOutEdge=3
                                                                            if newOutEdge == 1:
                                                                                if oe['sign'] == 1.0:
                                                                                    position1[index1]=positionPref1[-1]-oe['minDist']
                                                                                    if position1[index1] > 0:
                                                                                        positionPref1.append(position1[index1])
                                                                                        counterPref1.append(index)
                                                                                        countPref1 += 1
                                                                                    else:
                                                                                        print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 1')
                                                                                        print(position1[index1])
                                                                                        print(np.floor(space1/oe['minDist']))
                                                                                        print(timeBlocked1)
                                                                                        print(pref1Full)
                                                                                else:
                                                                                    position1[index1]=positionPref1[-1]+oe['minDist']
                                                                                    if position1[index1] < oe['length']:
                                                                                        positionPref1.append(position1[index1])
                                                                                        counterPref1.append(index)
                                                                                        countPref1 += 1
                                                                                    else:
                                                                                        print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 2')
                                                                                        print(position1[index1])
                                                                                        print(oe['length'])
                                                                                        print(np.floor(space1/oe['minDist']))
                                                                            elif newOutEdge == 2:
                                                                                if oe2['sign'] == 1.0:
                                                                                    position2[index2]=positionPref2[-1]-oe2['minDist']
                                                                                    if position2[index2] > 0:
                                                                                        positionPref2.append(position2[index2])
                                                                                        counterPref2.append(index)
                                                                                        countPref2 += 1
                                                                                    else:
                                                                                        print('WARNING PROPAGATE  RBC has been pushed outside SHOULD NOT HAPPEN 3')
                                                                                        print(position2[index2])
                                                                                        print(np.floor(space2/oe2['minDist']))
                                                                                        print(timeBlocked2)
                                                                                        print(pref2Full)
                                                                                else:
                                                                                    position2[index2]=positionPref2[-1]+oe2['minDist']
                                                                                    if position2[index2] < oe2['length']:
                                                                                        positionPref2.append(position2[index2])
                                                                                        counterPref2.append(index)
                                                                                        countPref2 += 1
                                                                                    else:
                                                                                        print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 4')
                                                                                        print(position2[index2])
                                                                                        print(oe2['length'])
                                                                                        print(np.floor(space2/oe2['minDist']))
                                                                            elif newOutEdge == 3:
                                                                                if oe3['sign'] == 1.0:
                                                                                    position3[index3]=positionPref3[-1]-oe3['minDist']
                                                                                    if position3[index3] > 0:
                                                                                        positionPref3.append(position3[index3])
                                                                                        counterPref3.append(index)
                                                                                        countPref3 += 1
                                                                                    else:
                                                                                        print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 5')
                                                                                        print(position3[index3])
                                                                                        print(np.floor(space3/oe3['minDist']))
                                                                                        print(timeBlocked3)
                                                                                        print(pref3Full)
                                                                                else:
                                                                                    position3[index3]=positionPref3[-1]+oe3['minDist']
                                                                                    if position3[index3] < oe3['length']:
                                                                                        positionPref3.append(position3[index3])
                                                                                        counterPref3.append(index)
                                                                                        countPref3 += 1
                                                                                    else:
                                                                                        print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 6')
                                                                                        print(position3[index3])
                                                                                        print(oe3['length'])
                                                                                        print(np.floor(space3/oe3['minDist']))
                                                                        #There is enough space in outEdge 3
                                                                        else:
                                                                            positionPref3.append(position3[index3])
                                                                            counterPref3.append(index)
                                                                            countPref3 += 1
                                                                    #No RBCs have been put in outEPref3 so far
                                                                    else:
                                                                        if oe3['sign'] == 1.0:
                                                                            #If there are are already RBCs present in outE3
                                                                            #Check that there is no overtaking of RBCs
                                                                            if len(oe3['rRBC']) > 0:
                                                                                if oe3['rRBC'][0]-position3[index3] < oe3['minDist']:
                                                                                    position3[index3]=oe3['rRBC'][0]-oe3['minDist']
                                                                            #There are no RBCs present in outE3
                                                                            else:
                                                                                #Avoid overshooting whole vessels
                                                                                if position3[index3] > oe3['length']:
                                                                                    position3[index3]=oe3['length']
                                                                        else:
                                                                            #If there are are already RBCs present in outE3
                                                                            #Check that there is no overtaking of RBCs
                                                                            if len(oe3['rRBC']) > 0:
                                                                                if position3[index3]-oe3['rRBC'][-1] < oe3['minDist']:
                                                                                    position3[index3]=oe3['rRBC'][-1]+oe3['minDist']
                                                                            else:
                                                                                #Avoid overshooting whole vessels
                                                                                if position3[index3] < 0:
                                                                                    position3[index3]=0
                                                                        positionPref3.append(position3[index3])
                                                                        counterPref3.append(index)
                                                                        countPref3 += 1
                                                                #There is no spcae in the third outEdge anymore
                                                                else:
                                                                    #Check if another RBCs still fits into the vessel
                                                                    space1 =  positionPref1[-1] if oe['sign'] == 1.0 \
                                                                        else oe['length']-positionPref1[-1]
                                                                    if np.floor(space1/oe['minDist']) > 1:
                                                                        timeBlocked1=(oe['minDist']-dist1)/oe['v']
                                                                    else:
                                                                        timeBlocked1=None
                                                                        pref1Full=1
                                                                    space2 =  positionPref2[-1] if oe2['sign'] == 1.0 \
                                                                        else oe2['length']-positionPref2[-1]
                                                                    if np.floor(space2/oe2['minDist']) > 1:
                                                                        timeBlocked2=(oe2['minDist']-dist2)/oe2['v']
                                                                    else:
                                                                        timeBlocked2=None
                                                                        pref2Full=1
                                                                    if pref1Full == 1 and pref2Full == 1:
                                                                        break
                                                                    #Define newOutEdge
                                                                    newOutEdge=0
                                                                    if timeBlocked1 == None:
                                                                        newOutEdge=2
                                                                    elif timeBlocked2 == None:
                                                                        newOutEdge=1
                                                                    else:
                                                                        if timeBlocked1 <= timeBlocked2:
                                                                            newOutEdge=1
                                                                        else:
                                                                            newOutEdge=2
                                                                    if newOutEdge == 1:
                                                                        if oe['sign'] == 1.0:
                                                                            position1[index1]=positionPref1[-1]-oe['minDist']
                                                                            if position1[index1] > 0:
                                                                                positionPref1.append(position1[index1])
                                                                                counterPref1.append(index)
                                                                                countPref1 += 1
                                                                            else:
                                                                                print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 7')
                                                                                print(position1[index1])
                                                                                print(np.floor(space1/oe['minDist']))
                                                                                print(timeBlocked1)
                                                                                print(pref1Full)
                                                                        else:
                                                                            position1[index1]=positionPref1[-1]+oe['minDist']
                                                                            if position1[index1] < oe['length']:
                                                                                positionPref1.append(position1[index1])
                                                                                counterPref1.append(index)
                                                                                countPref1 += 1
                                                                            else:
                                                                                print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 8')
                                                                                print(position1[index1])
                                                                                print(oe['length'])
                                                                                print(np.floor(space1/oe['minDist']))
                                                                    elif newOutEdge == 2:
                                                                        if oe2['sign'] == 1.0:
                                                                            position2[index2]=positionPref2[-1]-oe2['minDist']
                                                                            if position2[index2] > 0:
                                                                                positionPref2.append(position2[index2])
                                                                                counterPref2.append(index)
                                                                                countPref2 += 1
                                                                            else:
                                                                                print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 9')
                                                                                print(position2[index2])
                                                                                print(np.floor(space2/oe2['minDist']))
                                                                                print(timeBlocked2)
                                                                                print(pref2Full)
                                                                        else:
                                                                            position2[index2]=positionPref2[-1]+oe2['minDist']
                                                                            if position2[index2] < oe2['length']:
                                                                                positionPref2.append(position2[index2])
                                                                                counterPref2.append(index)
                                                                                countPref2 += 1
                                                                            else:
                                                                                print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 10')
                                                                                print(position2[index2])
                                                                                print(oe2['length'])
                                                                                print(np.floor(space2/oe2['minDist']))
                                                            #There is no third outEdge, therefore it is checked in which edge the RBC is blocked
                                                            #the shortest time
                                                            else:
                                                                #Check if another RBCs still fits into the vessel
                                                                space1 =  positionPref1[-1] if oe['sign'] == 1.0 \
                                                                    else oe['length']-positionPref1[-1]
                                                                if np.floor(space1/oe['minDist']) > 1:
                                                                    timeBlocked1=(oe['minDist']-dist1)/oe['v']
                                                                else:
                                                                    timeBlocked1=None
                                                                    pref1Full=1
                                                                space2 =  positionPref2[-1] if oe2['sign'] == 1.0 \
                                                                    else oe2['length']-positionPref2[-1]
                                                                if np.floor(space2/oe2['minDist']) > 1:
                                                                    timeBlocked2=(oe2['minDist']-dist2)/oe2['v']
                                                                else:
                                                                    timeBlocked2=None
                                                                    pref2Full=1
                                                                if pref1Full == 1 and pref2Full == 1:
                                                                    break
                                                                #Define newOutEdge
                                                                newOutEdge=0
                                                                if timeBlocked1 == None:
                                                                    newOutEdge=2
                                                                elif timeBlocked2 == None:
                                                                    newOutEdge=1
                                                                else:
                                                                    if timeBlocked1 <= timeBlocked2:
                                                                        newOutEdge=1
                                                                    else:
                                                                        newOutEdge=2
                                                                if newOutEdge == 1:
                                                                    if oe['sign'] == 1.0:
                                                                        position1[index1]=positionPref1[-1]-oe['minDist']
                                                                        if position1[index1] > 0:
                                                                            positionPref1.append(position1[index1])
                                                                            counterPref1.append(index)
                                                                            countPref1 += 1
                                                                        else:
                                                                            print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 11')
                                                                            print(position1[index1])
                                                                            print(np.floor(space1/oe['minDist']))
                                                                            print(timeBlocked1)
                                                                            print(pref1Full)
                                                                    else:
                                                                        position1[index1]=positionPref1[-1]+oe['minDist']
                                                                        if position1[index1] < oe['length']:
                                                                            positionPref1.append(position1[index1])
                                                                            counterPref1.append(index)
                                                                            countPref1 += 1
                                                                        else:
                                                                            print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 12')
                                                                            print(position1[index1])
                                                                            print(oe['length'])
                                                                            print(np.floor(space1/oe['minDist']))
                                                                elif newOutEdge == 2:
                                                                    if oe2['sign'] == 1.0:
                                                                        position2[index2]=positionPref2[-1]-oe2['minDist']
                                                                        if position2[index2] > 0:
                                                                            positionPref2.append(position2[index2])
                                                                            counterPref2.append(index)
                                                                            countPref2 += 1
                                                                        else:
                                                                            print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 13')
                                                                            print(position2[index2])
                                                                            print(np.floor(space2/oe2['minDist']))
                                                                            print(timeBlocked2)
                                                                            print(pref2Full)
                                                                    else:
                                                                        position2[index2]=positionPref2[-1]+oe2['minDist']
                                                                        if position2[index2] < oe2['length']:
                                                                            positionPref2.append(position2[index2])
                                                                            counterPref2.append(index)
                                                                            countPref2 += 1
                                                                        else:
                                                                            print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 14')
                                                                            print(position2[index2])
                                                                            print(oe2['length'])
                                                                            print(np.floor(space2/oe2['minDist']))
                                                        #there is enough space for the RBC in outEPref2
                                                        else:
                                                            positionPref2.append(position2[index2])
                                                            counterPref2.append(index)
                                                            countPref2 += 1
                                                    #no RBCs have been put in outEPref2 so far
                                                    else:
                                                        if oe2['sign'] == 1.0:
                                                            #If there are are already RBCs present in outE2
                                                            #Check that there is no overtaking of RBCs
                                                            if len(oe2['rRBC']) > 0:
                                                                if oe2['rRBC'][0]-position2[index2] < oe2['minDist']:
                                                                    position2[index2]=oe2['rRBC'][0]-oe2['minDist']
                                                            #There are no RBCs present in outE2
                                                            else:
                                                                #Avoid overshooting whole vessels
                                                                if position2[index2] > oe2['length']:
                                                                    position2[index2]=oe2['length']
                                                        else:
                                                            if len(oe2['rRBC']) > 0:
                                                                if position2[index2]-oe2['rRBC'][-1] < oe2['minDist']:
                                                                    position2[index2]=oe2['rRBC'][-1]+oe2['minDist']
                                                            else:
                                                                if position2[index2] < 0:
                                                                    position2[index2]=0
                                                        positionPref2.append(position2[index2])
                                                        counterPref2.append(index)
                                                        countPref2 += 1
                                                #There is no space in the second outEdge
                                                #Check if there is a third outEdge
                                                else:
                                                    if len(outEdges)>2:
                                                        #check if RBC still fits into outEPref3
                                                        if posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                                            #Check if RBCs have already been put into 
                                                            if positionPref3 != []:
                                                            #Check if distance to preceding RBC is big enough
                                                                dist3=positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                                    else position3[index3]-positionPref3[-1]
                                                                if dist3 < oe3['minDist']:
                                                                    #Check if another RBCs still fits into the vessel
                                                                    space1 =  positionPref1[-1] if oe['sign'] == 1.0 \
                                                                        else oe['length']-positionPref1[-1]
                                                                    if np.floor(space1/oe['minDist']) > 1:
                                                                        timeBlocked1=(oe['minDist']-dist1)/oe['v']
                                                                    else:
                                                                        timeBlocked1=None
                                                                        pref1Full=1
                                                                    space3 =  positionPref3[-1] if oe3['sign'] == 1.0 \
                                                                        else oe3['length']-positionPref3[-1]
                                                                    if np.floor(space3/oe3['minDist']) > 1:
                                                                        timeBlocked3=(oe3['minDist']-dist3)/oe3['v']
                                                                    else:
                                                                        timeBlocked3=None
                                                                        pref3Full=1
                                                                    if pref1Full == 1 and pref3Full == 1:
                                                                        break
                                                                    #Define newOutEdge
                                                                    newOutEdge=0
                                                                    if timeBlocked1 == None:
                                                                        newOutEdge=3
                                                                    elif timeBlocked3 == None:
                                                                        newOutEdge=1
                                                                    else:
                                                                        if timeBlocked1 <= timeBlocked3:
                                                                            newOutEdge=1
                                                                        else:
                                                                            newOutEdge=3
                                                                    if newOutEdge == 1:
                                                                        if oe['sign'] == 1.0:
                                                                            position1[index1]=positionPref1[-1]-oe['minDist']
                                                                            if position1[index1] > 0:
                                                                                positionPref1.append(position1[index1])
                                                                                counterPref1.append(index)
                                                                                countPref1 += 1
                                                                            else:
                                                                                print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN! 15')
                                                                                print(position1[index1])
                                                                                print(np.floor(space1/oe['minDist']))
                                                                                print(timeBlocked1)
                                                                                print(pref1Full)
                                                                        else:
                                                                            position1[index1]=positionPref1[-1]+oe['minDist']
                                                                            if position1[index1] < oe['length']:
                                                                                positionPref1.append(position1[index1])
                                                                                counterPref1.append(index)
                                                                                countPref1 += 1
                                                                            else:
                                                                                print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN! 16')
                                                                                print(position1[index1])
                                                                                print(oe['length'])
                                                                                print(np.floor(space1/oe['minDist']))
                                                                    elif newOutEdge == 3:
                                                                        if oe3['sign'] == 1.0:
                                                                            position3[index3]=positionPref3[-1]-oe3['minDist']
                                                                            if position3[index3] > 0:
                                                                                positionPref3.append(position3[index3])
                                                                                counterPref3.append(index)
                                                                                countPref3 += 1
                                                                            else:
                                                                                print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN! 17')
                                                                                print(position3[index3])
                                                                                print(np.floor(space3/oe3['minDist']))
                                                                                print(timeBlocked3)
                                                                                print(pref3Full)
                                                                        else:
                                                                            position3[index3]=positionPref3[-1]+oe3['minDist']
                                                                            if position3[index3] < oe3['length']:
                                                                                positionPref3.append(position3[index3])
                                                                                counterPref3.append(index)
                                                                                countPref3 += 1
                                                                            else:
                                                                                print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 18')
                                                                                print(position3[index3])
                                                                                print(oe3['length'])
                                                                                print(np.floor(space3/oe3['minDist']))
                                                                #There is enough space in outEdge 3
                                                                else:
                                                                    positionPref3.append(position3[index3])
                                                                    counterPref3.append(index)
                                                                    countPref3 += 1
                                                            #No RBCs have been put in outEPref3
                                                            else:
                                                                if oe3['sign'] == 1.0:
                                                                #If there are are already RBCs present in outE3
                                                                #Check that there is no overtaking of RBCs
                                                                    if len(oe3['rRBC']) > 0:
                                                                        if oe3['rRBC'][0]-position3[index3] < oe3['minDist']:
                                                                            position3[index3]=oe3['rRBC'][0]-oe3['minDist']
                                                                    #There are no RBCs present in outE3
                                                                    else:
                                                                        #Avoid overshooting whole vessels
                                                                        if position3[index3] > oe3['length']:
                                                                            position3[index3]=oe3['length']
                                                                else:
                                                                #If there are are already RBCs present in outE3
                                                                #Check that there is no overtaking of RBCs
                                                                    if len(oe3['rRBC']) > 0:
                                                                        if position3[index3]-oe3['rRBC'][-1] < oe3['minDist']:
                                                                            position3[index3]=oe3['rRBC'][-1]+oe3['minDist']
                                                                        else:
                                                                        #Avoid overshooting whole vessels
                                                                            if position3[index3] < 0:
                                                                                position3[index3]=0
                                                                positionPref3.append(position3[index3])
                                                                counterPref3.append(index)
                                                                countPref3 += 1
                                                        #There is no space in the third outEdge
                                                        else:
                                                        #RBC pushed backwards in Edge 1
                                                            if oe['sign'] == 1.0:
                                                                position1[index1]=positionPref1[-1]-oe['minDist']
                                                                if position1[index1] > 0:
                                                                    positionPref1.append(position1[index1])
                                                                    counterPref1.append(index)
                                                                    countPref1 += 1
                                                                else:
                                                                    pref1Full=1
                                                                    break
                                                            else:
                                                                position1[index1]=positionPref1[-1]+oe['minDist']
                                                                if position1[index1] < oe['length']:
                                                                    positionPref1.append(position1[index1])
                                                                    counterPref1.append(index)
                                                                    countPref1 += 1
                                                                else:
                                                                    pref1Full=1
                                                                    break
                                                    #There is no third outEdge
                                                    else:
                                                    #RBC pushed backwards in Edge 1
                                                        if oe['sign'] == 1.0:
                                                            position1[index1]=positionPref1[-1]-oe['minDist']
                                                            if position1[index1] > 0:
                                                                positionPref1.append(position1[index1])
                                                                counterPref1.append(index)
                                                                countPref1 += 1
                                                            else:
                                                                pref1Full=1
                                                                break
                                                        else:
                                                            position1[index1]=positionPref1[-1]+oe['minDist']
                                                            if position1[index1] < oe['length']:
                                                                positionPref1.append(position1[index1])
                                                                counterPref1.append(index)
                                                                countPref1 += 1
                                                            else:
                                                                pref1Full=1
                                                                break
                                            #If the RBC fits into outEPref1
                                            else:
                                                positionPref1.append(position1[index1])
                                                counterPref1.append(index)
                                                countPref1 += 1
                                        #There are not yet any new RBCs in outEdgePref
                                        else:
                                            if oe['sign'] == 1.0:
                                                #If there are are already RBCs present in outE1
                                                #Check that there is no overtaking of RBCs
                                                if len(oe['rRBC']) > 0:
                                                    if oe['rRBC'][0]-position1[index1] < oe['minDist']:
                                                        position1[index1]=oe['rRBC'][0]-oe['minDist']
                                                #There are no RBCs present in outE
                                                else:
                                                    #Avoid overshooting whole vessels
                                                    if position1[index1] > oe['length']:
                                                        position1[index1]=oe['length']
                                            else:
                                                if len(oe['rRBC']) > 0:
                                                    if position1[index1]-oe['rRBC'][-1] <oe['minDist']:
                                                        position1[index1]=oe['rRBC'][-1]+oe['minDist']
                                                else:
                                                    if position1[index1] < 0:
                                                        position1[index1]=0
                                            positionPref1.append(position1[index1])
                                            counterPref1.append(index)
                                            countPref1 += 1
                                    #The RBCs do not fit into the prefered outEdge anymore
                                    #Therefore they are either put in outEdge2 or outEdge3
                                    elif posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                        #Check if there are already new RBCs in outEPref2
                                        if positionPref2 != []:
                                            dist2=positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                                else position2[index2]-positionPref2[-1]
                                            if dist2 < oe2['minDist']:
                                                #Check if there is a third outEdge
                                                if len(outEdges)>2:
                                                    if posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                                        #Check if there are RBCs in the third outEdge
                                                        if positionPref3 != []:
                                                            dist3=positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                                else position3[index3]-positionPref3[-1]
                                                            #if there is not enough space in the third outEdge
                                                            #Check in which edge the RBC is blocked the shortest time
                                                            if dist3 < oe3['minDist']:
                                                                #Check if another RBCs still fits into the vessel
                                                                space2 =  positionPref2[-1] if oe2['sign'] == 1.0 \
                                                                    else oe2['length']-positionPref2[-1]
                                                                if np.floor(space2/oe2['minDist']) > 1:
                                                                    timeBlocked2=(oe2['minDist']-dist2)/oe2['v']
                                                                else:
                                                                    timeBlocked2=None
                                                                    pref2Full=1
                                                                space3 =  positionPref3[-1] if oe3['sign'] == 1.0 \
                                                                    else oe3['length']-positionPref3[-1]
                                                                if np.floor(space3/oe3['minDist']) > 1:
                                                                    timeBlocked3=(oe3['minDist']-dist3)/oe3['v']
                                                                else:
                                                                    timeBlocked3=None
                                                                    pref3Full=1
                                                                if pref2Full == 1 and pref3Full == 1:
                                                                    break
                                                                #Define newOutEdge
                                                                newOutEdge=0
                                                                if timeBlocked2 == None:
                                                                    newOutEdge=3
                                                                elif timeBlocked3 == None:
                                                                    newOutEdge=2
                                                                else:
                                                                    if timeBlocked2 <= timeBlocked3:
                                                                        newOutEdge=2
                                                                    else:
                                                                        newOutEdge=3
                                                                if newOutEdge == 2:
                                                                    if oe2['sign'] == 1.0:
                                                                        position2[index2]=positionPref2[-1]-oe2['minDist']
                                                                        if position2[index2] > 0:
                                                                            positionPref2.append(position2[index2])
                                                                            counterPref2.append(index)
                                                                            countPref2 += 1
                                                                        else:
                                                                            print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 19')
                                                                            print(position2[index2])
                                                                            print(np.floor(space2/oe2['minDist']))
                                                                            print(timeBlocked2)
                                                                            print(pref2Full)
                                                                    else:
                                                                        position2[index2]=positionPref2[-1]+oe2['minDist']
                                                                        if position2[index2] < oe2['length']:
                                                                            positionPref2.append(position2[index2])
                                                                            counterPref2.append(index)
                                                                            countPref2 += 1
                                                                        else:
                                                                            print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 20')
                                                                            print(position2[index2])
                                                                            print(oe2['length'])
                                                                            print(np.floor(space2/oe2['minDist']))
                                                                elif newOutEdge == 3:
                                                                    if oe3['sign'] == 1.0:
                                                                        position3[index3]=positionPref3[-1]-oe3['minDist']
                                                                        if position3[index3] > 0:
                                                                            positionPref3.append(position3[index3])
                                                                            counterPref3.append(index)
                                                                            countPref3 += 1
                                                                        else:
                                                                            print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 21')
                                                                            print(position3[index3])
                                                                            print(np.floor(space3/oe3['minDist']))
                                                                            print(timeBlocked3)
                                                                            print(pref3Full)
                                                                    else:
                                                                        position3[index3]=positionPref3[-1]+oe3['minDist']
                                                                        if position3[index3] < oe3['length']:
                                                                            positionPref3.append(position3[index3])
                                                                            counterPref3.append(index)
                                                                            countPref3 += 1
                                                                        else:
                                                                            print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 22')
                                                                            print(position3[index3])
                                                                            print(oe3['length'])
                                                                            print(np.floor(space3/oe3['minDist']))
                                                            #There is enough space in outEdge 3
                                                            else:
                                                                positionPref3.append(position3[index3])
                                                                counterPref3.append(index)
                                                                countPref3 += 1
                                                        #There are no RBCs in outEdgePref3
                                                        else:
                                                            #Check if RBC overshooted the vessel, or runs into the preceding one (which already is in the outEdge)
                                                            if oe3['sign'] == 1.0:
                                                                if len(oe3['rRBC']) > 0:
                                                                    if oe3['rRBC'][0]-position3[index3] < oe3['minDist']:
                                                                        position3[index3]=oe3['rRBC'][0]-oe3['minDist']
                                                                else:
                                                                    if position3[index3] > oe3['length']:
                                                                        position3[index3]=oe3['length']
                                                            else:
                                                                if len(oe3['rRBC']) > 0:
                                                                    if position3[index3]-oe3['rRBC'][-1] < oe3['minDist']:
                                                                        position3[index3]=oe3['rRBC'][-1]+oe3['minDist']
                                                                else:
                                                                    if position3[index3] < 0:
                                                                        position3[index3]=0
                                                            positionPref3.append(position3[index3])
                                                            counterPref3.append(index)
                                                            countPref3 += 1
                                                    else:
                                                    #There is no space in the third outEdge
                                                        if oe2['sign'] == 1.0:
                                                            position2[index2]=positionPref2[-1]-oe2['minDist']
                                                            if position2[index2] > 0:
                                                                positionPref2.append(position2[index2])
                                                                counterPref2.append(index)
                                                                countPref2 += 1
                                                            else:
                                                                pref2Full = 1
                                                                break
                                                        else:
                                                            position2[index2]=positionPref2[-1]+oe2['minDist']
                                                            if position2[index2] < oe2['length']:
                                                                positionPref2.append(position2[index2])
                                                                counterPref2.append(index)
                                                                countPref2 += 1
                                                            else:
                                                                pref2Full = 1
                                                                break
                                                #There is no third outEdge
                                                #The RBCs are pushed backwards such that there is no overlap
                                                else:
                                                    if oe2['sign'] == 1.0:
                                                        position2[index2]=positionPref2[-1]-oe2['minDist']
                                                        if position2[index2] > 0:
                                                            positionPref2.append(position2[index2])
                                                            counterPref2.append(index)
                                                            countPref2 += 1
                                                        else:
                                                            pref2Full = 1
                                                            break
                                                    else:
                                                        position2[index2]=positionPref2[-1]+oe2['minDist']
                                                        if position2[index2] < oe2['length']:
                                                            positionPref2.append(position2[index2])
                                                            counterPref2.append(index)
                                                            countPref2 += 1
                                                        else:
                                                            pref2Full = 1
                                                            break
                                            #There is enough space for the RBCs in the outEdge 2
                                            else:
                                                positionPref2.append(position2[index2])
                                                counterPref2.append(index)
                                                countPref2 += 1
                                        #No RBCs have been put into outEPref2 yet
                                        else:
                                            if oe2['sign'] == 1.0:
                                                if len(oe2['rRBC']) > 0:
                                                    if oe2['rRBC'][0]-position2[index2] < oe2['minDist']:
                                                        position2[index2]=oe2['rRBC'][0]-oe2['minDist']
                                                else:
                                                    if position2[index2] > oe2['length']:
                                                        position2[index2]=oe2['length']
                                            else:
                                                if len(oe2['rRBC']) > 0:
                                                    if position2[index2]-oe2['rRBC'][-1] < oe2['minDist']:
                                                        position2[index2]=oe2['rRBC'][-1]+oe2['minDist']
                                                else:
                                                    if position2[index2] < 0:
                                                        position2[index2]=0
                                            positionPref2.append(position2[index2])
                                            counterPref2.append(index)
                                            countPref2 += 1
                                    else:
                                        #Check if there is a third outEdge
                                        if len(outEdges)>2:
                                            #Check if there are RBCs in the third outEdge
                                            if posNoBifEventsPref3 > countPref3 and pref3Full == 0:
                                                if positionPref3 != []:
                                                    dist3=positionPref3[-1]-position3[index3] if oe3['sign'] == 1.0 \
                                                        else position3[index3]-positionPref3[-1]
                                                    #if there is not enough space in the third outEdge
                                                    if dist3 < oe3['minDist']:
                                                        #Push RBCs backwards to fit
                                                        if oe3['sign'] == 1.0:
                                                            position3[index3]=positionPref3[-1]-oe3['minDist']
                                                            if position3[index3] > 0:
                                                                positionPref3.append(position3[index3])
                                                                counterPref3.append(index)
                                                                countPref3 += 1
                                                            else:
                                                                pref3Full = 1
                                                                break
                                                        else: 
                                                            position3[index3]=positionPref3[-1]+oe3['minDist']
                                                            if position3[index3] < oe3['length']:
                                                                positionPref3.append(position3[index3])
                                                                counterPref3.append(index)
                                                                countPref3 += 1
                                                            else:
                                                                pref3Full = 1
                                                                break
                                                    #There is enough space in outEdge 3
                                                    else:
                                                        positionPref3.append(position3[index3])
                                                        counterPref3.append(index)
                                                        countPref3 += 1
                                                #No RBCs have been put in outEPref3 yet
                                                else:
                                                    if oe3['sign'] == 1.0:
                                                        if len(oe3['rRBC']) > 0:
                                                            if oe3['rRBC'][0]-position3[index3] < oe3['minDist']:
                                                                position3[index3]=oe3['rRBC'][0]-oe3['minDist']
                                                        else:
                                                            if position3[index3] > oe3['length']:
                                                                position3[index3]=oe3['length']
                                                    else:
                                                        if len(oe3['rRBC'])>0:
                                                            if position3[index3]-oe3['rRBC'][-1] < oe3['minDist']:
                                                                position3[index3]=oe3['rRBC'][-1]+oe3['minDist']
                                                        else:
                                                            if position3[index3] < 0:
                                                                position3[index3]=0
                                                    positionPref3.append(position3[index3])
                                                    counterPref3.append(index)
                                                    countPref3 += 1
                                            #No space in Pref3
                                            else:
                                                break
                                        #No third out Edge
                                        else:
                                            break
                                #Add rbcs to outEPref 
                                #It has been looped over all overshoot RBCs and the number of possible overshoots hase been corrected 
                                if len(outEdges) > 2:
                                    if countPref2+countPref1+countPref3 != overshootsNo:
                                        overshootsNo = countPref2+countPref1+countPref3
                                else:
                                    if countPref2+countPref1 != overshootsNo:
                                        overshootsNo = countPref2+countPref1
                            #add RBCs to outEPref1
                            nRBCint=oe['nRBCint']
                            for i in positionPref1:
                                nRBCint += self._nRBC_mean_inflow(i,oe,dt)
                            oe['nRBCint']=nRBCint
                            if oe['sign'] == 1.0:
                                oe['rRBC']=np.concatenate([positionPref1[::-1], oe['rRBC']])
                            else:
                                oe['rRBC']=np.concatenate([oe['rRBC'],positionPref1])
                            #Add rbcs to outEPref2       
                            nRBCint=oe2['nRBCint']
                            for i in positionPref2:
                                nRBCint += self._nRBC_mean_inflow(i,oe2,dt)
                            oe2['nRBCint']=nRBCint
                            if oe2['sign'] == 1.0:
                                oe2['rRBC']=np.concatenate([positionPref2[::-1], oe2['rRBC']])
                            else:
                                oe2['rRBC']=np.concatenate([oe2['rRBC'],positionPref2])
                            if len(outEdges) >2:
                            #Add rbcs to outEPref3       
                                nRBCint=oe3['nRBCint']
                                for i in positionPref3:
                                    nRBCint += self._nRBC_mean_inflow(i,oe3,dt)
                                oe3['nRBCint']=nRBCint
                                if oe3['sign'] == 1.0:
                                    oe3['rRBC']=np.concatenate([positionPref3[::-1], oe3['rRBC']])
                                else:
                                    oe3['rRBC']=np.concatenate([oe3['rRBC'],positionPref3])
                            #Remove RBCs from old Edge
                            if overshootsNo > 0:
                                nRBCint=e['nRBCint']
                                if e['sign'] == 1:
                                    for i in range(overshootsNo):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold[-1*(i+1)],e,dt)
                                else:
                                    for i in range(overshootsNo):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold[i],e,dt)
                                if sign == 1.0:
                                    e['rRBC']=e['rRBC'][:-overshootsNo]
                                else:
                                    e['rRBC']=e['rRBC'][overshootsNo::]
                                nRBCint += len(e['rRBC'])
                                e['nRBCint']=nRBCint
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        noStuckRBCs=len(bifRBCsIndex)-overshootsNo
                        for i in range(noStuckRBCs):
                            index=-1*(i+1) if sign == 1.0 else i
                            e['rRBC'][index]=e['length']-i*e['minDist'] if sign == 1.0 else 0+i*e['minDist']
                        #Recheck if the distance between the newly introduces RBCs is still big enough 
                        if len(e['rRBC']) >1:
                            moved = 0
                            count = 0
                            if sign == 1.0:
                                for i in range(-1,-1*(len(e['rRBC'])),-1):
                                    index=i-1
                                    if e['rRBC'][i] < e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]-e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs and moved == 0:
                                        break
                            else:
                                for i in range(len(e['rRBC'])-1):
                                    index=i+1
                                    if e['rRBC'][i] > e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]+e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs+1 and moved == 0:
                                        break
    #-------------------------------------------------------------------------------------------
                #if vertex is convergent vertex
                    elif G.vs[vi]['vType'] == 4:
                        bifRBCsIndex1=bifRBCsIndex
                        noBifEvents1=noBifEvents
                        outE=G.vs[vi]['outflowE'][0]
                        oe = G.es[outE]
                        inflowEdges=G.vs[vi]['inflowE']
                        k=0
                        for i in inflowEdges:
                            if i == e.index:
                                inE1=e.index
                            else:
                                if k == 0:
                                    inE2=i
                                    k = 1
                                else:
                                    inE3=i
                        e2=G.es[inE2]
                        #Move RBCs in second inEdge (if that has not been done already)
                        if inE2 not in convEdges2:
                            convEdges2.append(inE2)
                            #If RBCs are present move all RBCs in inEdge2
                            if len(e2['rRBC']) > 0:
                                rRBCold2=e2['rRBC']
                                e2['rRBC'] = e2['rRBC'] + e2['v'] * dt * e2['sign']
                                bifRBCsIndex2=[]
                                nRBC2=len(e2['rRBC'])
                                if e2['sign'] == 1.0:
                                    if e2['rRBC'][-1] > e2['length']:
                                        for i,j in enumerate(e2['rRBC'][::-1]):
                                            if j > e2['length']:
                                                bifRBCsIndex2.append(nRBC2-1-i)
                                            else:
                                                break
                                    bifRBCsIndex2=bifRBCsIndex2[::-1]
                                else:
                                    if e2['rRBC'][0] < 0:
                                        for i,j in enumerate(e2['rRBC']):
                                            if j < 0:
                                                bifRBCsIndex2.append(i)
                                            else:
                                                break
                                noBifEvents2=len(bifRBCsIndex2)
                            else:
                                bifRBCsIndex2=[]
                                noBifEvents2=0
                            sign2=e2['sign']
                        else:
                            noBifEvents2=0
                            bifRBCsIndex2=[]
                            sign2=e2['sign']
                        #Check if there is a third inEdge
                        if len(inflowEdges) > 2:
                            e3=G.es[inE3]
                            if inE3 not in convEdges2:
                                convEdges2.append(inE3)
                                #If RBCs are present move all RBCs in inEdge3
                                if len(e3['rRBC']) > 0:
                                    rRBCold3=e3['rRBC']
                                    e3['rRBC'] = e3['rRBC'] + e3['v'] * dt * e3['sign']
                                    bifRBCsIndex3=[]
                                    nRBC3=len(e3['rRBC'])
                                    if e3['sign'] == 1.0:
                                        if e3['rRBC'][-1] > e3['length']:
                                            for i,j in enumerate(e3['rRBC'][::-1]):
                                                if j > e3['length']:
                                                    bifRBCsIndex3.append(nRBC3-1-i)
                                                else:
                                                    break
                                        bifRBCsIndex3=bifRBCsIndex3[::-1]
                                    else:
                                        if e3['rRBC'][0] < 0:
                                            for i,j in enumerate(e3['rRBC']):
                                                if j < 0:
                                                    bifRBCsIndex3.append(i)
                                                else:
                                                    break
                                    noBifEvents3=len(bifRBCsIndex3)
                                else:
                                    bifRBCsIndex3=[]
                                    noBifEvents3=0
                                sign3=e3['sign']
                            else:
                                bifRBCsIndex3=[]
                                sign3=e3['sign']
                                noBifEvents3=0
                        else:
                            bifRBCsIndex3=[]
                            noBifEvents3=0
                        #Calculate distance to first RBC in outEdge
                        if len(G.es[outE]['rRBC']) > 0:
                            distToFirst=G.es[outE]['rRBC'][0] if G.es[outE]['sign'] == 1.0 else G.es[outE]['length']-G.es[outE]['rRBC'][-1]
                        else:
                            distToFirst=G.es[outE]['length']
                        posNoBifEvents=int(np.floor(distToFirst/G.es[outE]['minDist']))
                        if posNoBifEvents + len(G.es[outE]['rRBC']) > G.es[outE]['nMax']:
                            posNoBifEvents = G.es[outE]['nMax'] - len(G.es[outE]['rRBC'])
                        #If bifurcations are possible check how many overshoots there are at the inEdges
                        if posNoBifEvents > 0:
                            #overshootDist starts with the RBC which overshoots the least
                            overshootDist1=[e['rRBC'][bifRBCsIndex1]-[e['length']]*noBifEvents1 if sign == 1.0
                                else [0]*noBifEvents1-e['rRBC'][bifRBCsIndex1]][0]
                            if sign != 1.0:
                                overshootDist1 = overshootDist1[::-1]
                            overshootTime1=np.array(overshootDist1 / ([e['v']]*noBifEvents1))
                            dummy1=np.array([1]*len(overshootTime1))
                            if noBifEvents2 > 0:
                                #overshootDist starts with the RBC which overshoots the least
                                overshootDist2=[e2['rRBC'][bifRBCsIndex2]-[e2['length']]*noBifEvents2 if sign2 == 1.0
                                    else [0]*noBifEvents2-e2['rRBC'][bifRBCsIndex2]][0]
                                if sign2 != 1.0:
                                    overshootDist2 = overshootDist2[::-1]
                                overshootTime2=np.array(overshootDist2)/ np.array([e2['v']]*noBifEvents2)
                                dummy2=np.array([2]*len(overshootTime2))
                            else:
                                overshootDist2=[]
                                overshootTime2=[]
                                dummy2=[]
                            if len(inflowEdges) > 2:
                                if noBifEvents3 > 0:
                                    overshootDist3=[e3['rRBC'][bifRBCsIndex3]-[e3['length']]*noBifEvents3 if sign3 == 1.0
                                        else [0]*noBifEvents3-e3['rRBC'][bifRBCsIndex3]][0]
                                    if sign3 != 1.0:
                                        overshootDist3 = overshootDist3[::-1]
                                    overshootTime3=np.array(overshootDist3)/ np.array([e3['v']]*noBifEvents3)
                                    dummy3=np.array([3]*len(overshootTime3))
                                else:
                                    overshootDist3=[]
                                    overshootTime3=[]
                                    dummy3=[]
                            else:
                                overshootDist3=[]
                                overshootTime3=[]
                                dummy3=[]
                            #Define which RBC arrive first, second, .. at convergent bifurcation
                            overshootTimes=zip(np.concatenate([overshootTime1,overshootTime2,overshootTime3]),np.concatenate([dummy1,dummy2,dummy3]))
                            overshootTimes.sort()
                            overshootTime=[]
                            inEdge=[]
                            count1=0
                            count2=0
                            count3=0
                            if posNoBifEvents > len(overshootTimes):
                                overshootsNo=int(len(overshootTimes))
                            else:
                                overshootsNo=int(posNoBifEvents)
                            #position rbcs based on when they appear at bifurcation
                            for i in range(-1*overshootsNo,0):
                                overshootTime.append(overshootTimes[i][0])
                                inEdge.append(overshootTimes[i][1])
                            #Numbers of RBCs from corresponding inEdge
                            count1=inEdge.count(1)
                            count2=inEdge.count(2)
                            count3=inEdge.count(3)
                            #position starts with least overshooting RBC and ends with highest overshooting RBC
                            position=np.array(overshootTime)*np.array([G.es[outE]['v']]*overshootsNo)
                            #Check if RBCs are to close to each other
                            #Check if the RBCs runs into an old one in the vessel
                            #(only position of the leading RBC is changed)
                            if len(G.es[outE]['rRBC']) > 0:
                                if G.es[outE]['sign'] == 1.0:
                                    if position[-1] > G.es[outE]['rRBC'][0]-G.es[outE]['minDist']:
                                        position[-1]=G.es[outE]['rRBC'][0]-G.es[outE]['minDist']
                                else:
                                    if G.es[outE]['length']-position[-1] < G.es[outE]['rRBC'][-1]+G.es[outE]['minDist']:
                                        position[-1]=G.es[outE]['length']-(G.es[outE]['rRBC'][-1]+G.es[outE]['minDist'])
                            else:
                                #Check if the RBCs overshooted the vessel
                                if position[-1] > G.es[outE]['length']:
                                    position[-1]=G.es[outE]['length']
                            #Position of the following RBCs is changed, such that they do not overlap
                            for i in range(-1,-1*(count1+count2+count3),-1):
                                if position[i]-position[i-1] < G.es[outE]['minDist'] or \
                                    position[i-1] > position[i]:
                                    position[i-1]=position[i]-G.es[outE]['minDist']
                            #if first RBC did not yet move enough less than the possible no of RBCs fit into the outEdge
                            for i in range(count1+count2+count3):
                                if position[i] < 0:
                                    if inEdge[i] == 1:
                                        count1 += -1
                                    elif inEdge[i] == 2:
                                        count2 += -1
                                    elif inEdge[i] == 3:
                                        count3 += -1
                                else:
                                    break
                            if len(position) != len(position[i::]):
                                position=position[i::]
                                inEdge=inEdge[i::]
                            #Add rbcs to outE
                            if oe['sign'] == 1.0:
                                oe['rRBC']=np.concatenate([position, oe['rRBC']])
                            else:
                                position = [oe['length']]*len(position) - position[::-1]
                                oe['rRBC']=np.concatenate([oe['rRBC'],position])
                            nRBCint=oe['nRBCint']
                            for i in position:
                                nRBCint += self._nRBC_mean_inflow(i,oe,dt)
                            oe['nRBCint']=nRBCint
                            #Remove RBCs from old Edge 1
                            if count1 > 0:
                                nRBCint=e['nRBCint']
                                if e['sign'] == 1:
                                    for i in range(count1):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold[-1*(i+1)],e,dt)
                                else:
                                    for i in range(count1):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold[i],e,dt)
                                if sign == 1.0:
                                    e['rRBC']=e['rRBC'][:-count1]
                                else:
                                    e['rRBC']=e['rRBC'][count1::]
                                nRBCint += len(e['rRBC'])
                                e['nRBCint']=nRBCint
                            if noBifEvents2 > 0 and count2 > 0:
                                nRBCint=e2['nRBCint']
                                if e2['sign'] == 1:
                                    for i in range(count2):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold2[-1*(i+1)],e2,dt)
                                else:
                                    for i in range(count2):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold2[i],e2,dt)
                                #Remove RBCs from old Edge 2
                                if sign2 == 1.0:
                                    e2['rRBC']=e2['rRBC'][:-count2]
                                else:
                                    e2['rRBC']=e2['rRBC'][count2::]
                                nRBCint += len(e2['rRBC'])
                                e2['nRBCint']=nRBCint
                            if len(inflowEdges) > 2:
                                if noBifEvents3 > 0 and count3 > 0:
                                    nRBCint=e3['nRBCint']
                                    if e3['sign'] == 1:
                                        for i in range(count3):
                                            nRBCint += self._nRBC_mean_outflow(rRBCold3[-1*(i+1)],e3,dt)
                                    else:
                                        for i in range(count3):
                                            nRBCint += self._nRBC_mean_outflow(rRBCold3[i],e3,dt)
                                    #Remove RBCs from old Edge 3
                                    if sign3 == 1.0:
                                        e3['rRBC']=e3['rRBC'][:-count3]
                                    else:
                                        e3['rRBC']=e3['rRBC'][count3::]
                                    nRBCint += len(e3['nRBCint'])
                                    e3['nRBCint']=nRBCint
                            overshootsNo = count1 + count2 + count3
                        else:
                            count1=0
                            count2=0
                            count3=0
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        #InEdge 1
                        noStuckRBCs1=len(bifRBCsIndex1)-count1
                        for i in range(noStuckRBCs1):
                            index=-1*(i+1) if sign == 1.0 else i
                            e['rRBC'][index]=e['length']-i*e['minDist'] if sign == 1.0 else 0+i*e['minDist']
                        #Recheck if the distance between the newly introduces RBCs is still big enough 
                        if len(e['rRBC']) >1.0:
                            moved = 0
                            count = 0
                            if sign == 1.0:
                                for i in range(-1,-1*(len(e['rRBC'])),-1):
                                    index=i-1
                                    if e['rRBC'][i] < e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]-e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs1 and moved == 0:
                                        break
                            else:
                                for i in range(len(e['rRBC'])-1):
                                    index=i+1
                                    if e['rRBC'][i] > e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]+e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs1+1 and moved == 0:
                                        break
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        #InEdge 2
                        noStuckRBCs2=len(bifRBCsIndex2)-count2
                        for i in range(noStuckRBCs2):
                            index=-1*(i+1) if sign2 == 1.0 else i
                            e2['rRBC'][index]=e2['length']-i*e2['minDist'] if sign2 == 1.0 else 0+i*e2['minDist']
                        #Recheck if the distance between the newly introduces RBCs is still big enough 
                        if len(e2['rRBC']) >1:
                            moved = 0
                            count = 0
                            if sign2 == 1.0:
                                for i in range(-1,-1*(len(e2['rRBC'])),-1):
                                    index=i-1
                                    if e2['rRBC'][i] < e2['rRBC'][index] or abs(e2['rRBC'][i]-e2['rRBC'][index]) < e2['minDist']:
                                        e2['rRBC'][index]=e2['rRBC'][i]-e2['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs2 and moved == 0:
                                        break
                            else:
                                for i in range(len(e2['rRBC'])-1):
                                    index=i+1
                                    if e2['rRBC'][i] > e2['rRBC'][index] or abs(e2['rRBC'][i]-e2['rRBC'][index]) < e2['minDist']:
                                        e2['rRBC'][index]=e2['rRBC'][i]+e2['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs2+1 and moved == 0:
                                        break
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        #InEdge 3
                        if len(inflowEdges) > 2:
                            noStuckRBCs3=len(bifRBCsIndex3)-count3
                            for i in range(noStuckRBCs3):
                                index=-1*(i+1) if sign3 == 1.0 else i
                                e3['rRBC'][index]=e3['length']-i*e3['minDist'] if sign3 == 1.0 else 0+i*e3['minDist']
                            #Recheck if the distance between the newly introduces RBCs is still big enough 
                            if len(e3['rRBC']) >1:
                                moved = 0
                                count = 0
                                if sign3 == 1.0:
                                    for i in range(-1,-1*(len(e3['rRBC'])),-1):
                                        index=i-1
                                        if e3['rRBC'][i] < e3['rRBC'][index] or abs(e3['rRBC'][i]-e3['rRBC'][index]) < e3['minDist']:
                                            e3['rRBC'][index]=e3['rRBC'][i]-e3['minDist']
                                            moved = 1
                                        else:
                                            moved = 0
                                        count += 1
                                        if count >= noStuckRBCs3 and moved == 0:
                                            break
                                else:
                                    for i in range(len(e3['rRBC'])-1):
                                        index=i+1
                                        if e3['rRBC'][i] > e3['rRBC'][index] or abs(e3['rRBC'][i]-e3['rRBC'][index]) < e3['minDist']:
                                            e3['rRBC'][index]=e3['rRBC'][i]+e3['minDist']
                                            moved = 1
                                        else:
                                            moved = 0
                                        count += 1
                                        if count >= noStuckRBCs3+1 and moved == 0:
                                            break
         #------------------------------------------------------------------------------------------
                    #if vertex is double connecting vertex
                    elif G.vs[vi]['vType'] == 6:
                        bifRBCsIndex1=bifRBCsIndex
                        noBifEvents1=noBifEvents
                        outE=G.vs[vi]['outflowE'][0]
                        inflowEdges=G.vs[vi]['inflowE']
                        for i in inflowEdges:
                            if i == e.index:
                                inE1=e.index
                            else:
                                inE2=i
                        e2=G.es[inE2]
                        if inE2 not in convEdges2:
                            convEdges2.append(inE2)
                            #If RBCs are present move all RBCs in inEdge2
                            if len(e2['rRBC']) > 0:
                                rRBCold2=e2['rRBC']
                                e2['rRBC'] = e2['rRBC'] + e2['v'] * dt * e2['sign']
                                bifRBCsIndex2=[]
                                nRBC2=len(e2['rRBC'])
                                if e2['sign'] == 1.0:
                                    if e2['rRBC'][-1] > e2['length']:
                                        for i,j in enumerate(e2['rRBC'][::-1]):
                                            if j > e2['length']:
                                                bifRBCsIndex2.append(nRBC2-1-i)
                                            else:
                                                break
                                    bifRBCsIndex2=bifRBCsIndex2[::-1]
                                else:
                                    if e2['rRBC'][0] < 0:
                                        for i,j in enumerate(e2['rRBC']):
                                            if j < 0:
                                                bifRBCsIndex2.append(i)
                                            else:
                                                break
                                noBifEvents2=len(bifRBCsIndex2)
                            else:
                                noBifEvents2=0
                                bifRBCsIndex2=[]
                        else:
                            bifRBCsIndex2=[]
                            noBifEvents2=0
                        sign2=e2['sign']
                        #Define outEdges
                        outEdges=G.vs[vi]['outflowE']
                        outE=outEdges[0]
                        outE2=outEdges[1]
                        #Differ between capillaries and non-capillaries
                        if G.vs[vi]['isCap']:
                            nonCap = 0
                            preferenceList = [x[1] for x in sorted(zip(np.array(G.es[outEdges]['flow'])/np.array(G.es[outEdges]['crosssection']), outEdges), reverse=True)]
                        else:
                            nonCap = 1
                            preferenceList = [x[1] for x in sorted(zip(G.es[outEdges]['flow'], outEdges), reverse=True)]
                            ratio1 = Physiol.phase_separation_effect(G.es[preferenceList[0]]['flow']/np.sum(G.es[outEdges]['flow']), \
                                G.es[preferenceList[0]]['diameter'],G.es[preferenceList[1]]['diameter'],e['diameter'],e['htd'])
                            ratio2 = 1.0 -  ratio1
                        #Define prefered OutEdges
                        outEPref=preferenceList[0]
                        outEPref2=preferenceList[1]
                        oe=G.es[outEPref]
                        oe2=G.es[outEPref2]
                        #Calculate distance to first RBC
                        if len(oe['rRBC']) > 0:
                            distToFirst=oe['rRBC'][0] if oe['sign'] == 1.0 else oe['length']-oe['rRBC'][-1]
                        else:
                            distToFirst=oe['length']
                        if len(oe2['rRBC']) > 0:
                            distToFirst2=oe2['rRBC'][0] if oe2['sign'] == 1.0 else oe2['length']-oe2['rRBC'][-1]
                        else:
                            distToFirst2=oe2['length']
                        #Check how many RBCs are allowed by nMax
                        posNoBifEventsPref=np.floor(distToFirst/oe['minDist'])
                        if posNoBifEventsPref + len(oe['rRBC']) > oe['nMax']:
                            posNoBifEventsPref = oe['nMax'] - len(oe['rRBC'])
                        posNoBifEventsPref2=np.floor(distToFirst2/oe2['minDist'])
                        if posNoBifEventsPref2 + len(oe2['rRBC']) > oe2['nMax']:
                            posNoBifEventsPref2 = oe2['nMax'] - len(oe2['rRBC'])
                        #Check how many RBCs fit into the new Vessel
                        posNoBifEvents=int(posNoBifEventsPref+posNoBifEventsPref2)
                        noBifEvents = noBifEvents1 + noBifEvents2
                        if nonCap:
                            overshootsNo1 = np.floor(ratio1 * noBifEvents)
                            overshootsNo2 = noBifEvents - overshootsNo1
                            if overshootsNo1 > posNoBifEventsPref:
                                overshootsNo1 = posNoBifEventsPref
                            if overshootsNo2 > posNoBifEventsPref2:
                                overshootsNo2 = posNoBifEventsPref2
                            overshootsNo = int(overshootsNo1 + overshootsNo2)
                            posNoBifEvents = overshootsNo
                        #Calculate number of bifEvents
                        #If bifurcations are possible check how many overshoots there are at the inEdges
                        if posNoBifEvents > 0:
                            overshootDist1=[e['rRBC'][bifRBCsIndex1]-[e['length']]*noBifEvents1 if sign == 1.0
                                else [0]*noBifEvents1-e['rRBC'][bifRBCsIndex1]][0]
                            if sign != 1.0:
                                overshootDist1 = overshootDist1[::-1]
                            overshootTime1=np.array(overshootDist1 / ([e['v']]*noBifEvents1))
                            dummy1=np.array([1]*len(overshootTime1))
                            if noBifEvents2 > 0:
                                overshootDist2=[e2['rRBC'][bifRBCsIndex2]-[e2['length']]*noBifEvents2 if sign2 == 1.0
                                    else [0]*noBifEvents2-e2['rRBC'][bifRBCsIndex2]][0]
                                if sign2 != 1.0:
                                    overshootDist2 = overshootDist2[::-1]
                                overshootTime2=np.array(overshootDist2)/ np.array([e2['v']]*noBifEvents2)
                                dummy2=np.array([2]*len(overshootTime2))
                            else:
                                overshootDist2=[]
                                overshootTime2=[]
                                dummy2=[]
                            overshootTimes=zip(np.concatenate([overshootTime1,overshootTime2]),np.concatenate([dummy1,dummy2]))
                            overshootTimes.sort()
                            overshootTime=[]
                            inEdge=[]
                            #Count RBCs moving from inEdge1 and inEdge2
                            if posNoBifEvents > len(overshootTimes):
                                overshootsNo=int(len(overshootTimes))
                            else:
                                overshootsNo=int(posNoBifEvents)
                            for i in range(-1*overshootsNo,0):
                                overshootTime.append(overshootTimes[i][0])
                                inEdge.append(overshootTimes[i][1])
                            if oe['sign'] == 1.0:
                                position1=np.array(overshootTime)*np.array([oe['v']]*overshootsNo)
                            else:
                                position1=np.array([oe['length']]*overshootsNo)-np.array(overshootTime[::-1])* \
                                    np.array([oe['v']]*overshootsNo)
                            if oe2['sign'] == 1.0:
                                position2=np.array(overshootTime)*np.array([oe2['v']]*overshootsNo)
                            else:
                                position2=np.array([oe2['length']]*overshootsNo)-np.array(overshootTime[::-1])* \
                                    np.array([oe2['v']]*overshootsNo)
                            if nonCap:
                                countNo1=0
                                countNo2=0
                                count1 = 0
                                count2 = 0
                                inEPref1=[]
                                inEPref2=[]
                                indexPref1=[]
                                indexPref2=[]
                                positionPref2=[]
                                positionPref1=[]
                                last=2
                                for i in range(overshootsNo):
                                    index=-1*(i+1)
                                    index1=-1*(i+1) if oe['sign'] == 1.0 else i
                                    index2=-1*(i+1) if oe2['sign'] == 1.0 else i
                                    if last == 2:
                                        if countNo1 < overshootsNo1:
                                            if positionPref1 == []:
                                                if len(oe['rRBC']) > 0:
                                                    if oe['sign'] == 1:
                                                        if position1[index1] > oe['rRBC'][0]-oe['minDist']:
                                                            positionPref1.append(oe['rRBC'][0]-oe['minDist'])
                                                        else:
                                                            positionPref1.append(position1[index1])
                                                    else:
                                                        if position1[index1] < oe['rRBC'][-1]+oe['minDist']:
                                                            positionPref1.append(oe['rRBC'][-1]+oe['minDist'])
                                                        else:
                                                            positionPref1.append(position1[index1])
                                                else:
                                                    if oe['sign'] == 1:
                                                        if position1[index1] > oe['length']:
                                                            positionPref1.append(oe['length'])
                                                        else:
                                                            positionPref1.append(position1[index1])
                                                    else:
                                                        if position1[index1] < 0:
                                                            positionPref1.append(0)
                                                        else:
                                                            positionPref1.append(position1[index1])
                                            else:
                                                positionPref1.append(position1[index1])
                                            inEPref1.append(inEdge[index])
                                            indexPref1.append(i)
                                            countNo1 += 1
                                            last = 1
                                            if inEdge[index] == 1:
                                                count1 += 1
                                            else:
                                                count2 += 1
                                        else: 
                                            if countNo2 < overshootsNo2:
                                                if positionPref2 == []:
                                                    if len(oe2['rRBC']) > 0:
                                                        if oe2['sign'] == 1:
                                                            if position2[index2] > oe2['rRBC'][0]-oe2['minDist']:
                                                                positionPref2.append(oe2['rRBC'][0]-oe2['minDist'])
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                        else:
                                                            if position2[index2] < oe2['rRBC'][-1]+oe2['minDist']:
                                                                positionPref2.append(oe2['rRBC'][-1]+oe2['minDist'])
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                    else:
                                                        if oe2['sign'] == 1:
                                                            if position2[index2] > oe2['length']:
                                                                positionPref2.append(oe2['length'])
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                        else:
                                                            if position2[index2] < 0:
                                                                positionPref2.append(0)
                                                            else:
                                                                positionPref2.append(position2[index2])
                                                else:
                                                    positionPref2.append(position2[index2])
                                                inEPref2.append(inEdge[index])
                                                indexPref2.append(i)
                                                countNo2 += 1
                                                last = 2
                                                if inEdge[index] == 1:
                                                    count1 += 1
                                                else:
                                                    count2 += 1
                                            else:
                                                print('BIGERROR all overshootRBCS should fit')
                                    elif last == 1:
                                        if countNo2 < overshootsNo2:
                                            if positionPref2 == []:
                                                if len(oe2['rRBC']) > 0:
                                                    if oe2['sign'] == 1:
                                                        if position2[index2] > oe2['rRBC'][0]-oe2['minDist']:
                                                            positionPref2.append(oe2['rRBC'][0]-oe2['minDist'])
                                                        else:
                                                            positionPref2.append(position2[index2])
                                                    else:
                                                        if position2[index2] < oe2['rRBC'][-1]+oe2['minDist']:
                                                            positionPref2.append(oe2['rRBC'][-1]+oe2['minDist'])
                                                        else:
                                                            positionPref2.append(position2[index2])
                                                else:
                                                    if oe2['sign'] == 1:
                                                        if position2[index2] > oe2['length']:
                                                            positionPref2.append(oe2['length'])
                                                        else:
                                                            positionPref2.append(position2[index2])
                                                    else:
                                                        if position2[index2] < 0:
                                                            positionPref2.append(0)
                                                        else:
                                                            positionPref2.append(position2[index2])
                                            else:
                                                positionPref2.append(position2[index2])
                                            inEPref2.append(inEdge[index])
                                            indexPref2.append(i)
                                            countNo2 += 1
                                            last = 2
                                            if inEdge[index] == 1:
                                                count1 += 1
                                            else:
                                                count2 += 1
                                        else: 
                                            if countNo1 < overshootsNo1:
                                                if positionPref1 == []:
                                                    if len(oe['rRBC']) > 0:
                                                        if oe['sign'] == 1:
                                                            if position1[index1] > oe['rRBC'][0]-oe['minDist']:
                                                                positionPref1.append(oe['rRBC'][0]-oe['minDist'])
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                        else:
                                                            if position1[index1] < oe['rRBC'][-1]+oe['minDist']:
                                                                positionPref1.append(oe['rRBC'][-1]+oe['minDist'])
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                    else:
                                                        if oe['sign'] == 1:
                                                            if position1[index1] > oe['length']:
                                                                positionPref1.append(oe['length'])
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                        else:
                                                            if position1[index1] < 0:
                                                                positionPref1.append(0)
                                                            else:
                                                                positionPref1.append(position1[index1])
                                                else:
                                                    positionPref1.append(position1[index1])
                                                inEPref1.append(inEdge[index])
                                                indexPref1.append(i)
                                                countNo1 += 1
                                                last = 1
                                                if inEdge[index] == 1:
                                                    count1 += 1
                                                else:
                                                    count2 += 1
                                            else:
                                                print('BIGERROR all overshootRBCS should fit')
                                    if last == 1:
                                        if len(positionPref1) >= 2:
                                            if oe['sign'] == 1:
                                                if positionPref1[-1] > positionPref1[-2] or positionPref1[-2]-positionPref1[-1] < oe['minDist']-eps:
                                                    positionPref1[-1] = positionPref1[-2] - oe['minDist']
                                            else:
                                                if positionPref1[-1] < positionPref1[-2] or positionPref1[-1]-positionPref1[-2] < oe['minDist']-eps:
                                                    positionPref1[-1] = positionPref1[-2] + oe['minDist']
                                    elif last == 2:
                                        if len(positionPref2) >= 2:
                                            if oe2['sign'] == 1:
                                                if positionPref2[-1] > positionPref2[-2] or positionPref2[-2] - positionPref2[-1] < oe2['minDist']-eps:
                                                    positionPref2[-1] = positionPref2[-2] - oe2['minDist']
                                            else:
                                                if positionPref2[-1] < positionPref2[-2] or positionPref2[-1] - positionPref2[-2] < oe2['minDist']-eps:
                                                    positionPref2[-1] = positionPref2[-2] + oe2['minDist']
                                if positionPref1 != []:
                                    if oe['sign'] == 1:
                                        if positionPref1[-1] < 0:
                                            positionPref1[-1] = 0
                                            for i in range(-1,-1*(len(positionPref1)),-1):
                                                if positionPref1[i-1]-positionPref1[i] < oe['minDist'] - eps:
                                                    positionPref1[i-1]=positionPref1[i] + oe['minDist']
                                                else:
                                                    break
                                            if len(oe['rRBC']) > 0:
                                                if positionPref1[0] > oe['rRBC'][0]:
                                                    print('BIGERROR should not happen 1')
                                                    print(positionPref1[0])
                                                    print(oe['rRBC'][0])
                                        if positionPref1[0] > oe['length']:
                                            positionPref1[0] = oe['length']
                                            for i in range(len(positionPref1)-1):
                                                if positionPref1[i]-positionPref1[i+1] < oe['minDist'] - eps:
                                                    positionPref1[i+1]=positionPref1[i] - oe['minDist']
                                                else:
                                                    break
                                            if positionPref1[-1] < 0:
                                                print('BIGERROR should not happen 1_2')
                                                print(positionPref1[-1])
                                    else:
                                        if positionPref1[-1] > oe['length']:
                                            positionPref1[-1] = oe['length']
                                            for i in range(-1,-1*(len(positionPref1)),-1):
                                                if positionPref1[i]-positionPref1[i-1] < oe['minDist'] + eps:
                                                    positionPref1[i-1]=positionPref1[i] - oe['minDist']
                                                else:
                                                    break
                                            if len(oe['rRBC']) > 0:
                                                if positionPref1[0] < oe['rRBC'][-1]:
                                                    print('BIGERROR should not happen 2')
                                                    print(positionPref1[0])
                                                    print(oe['rRBC'][-1])
                                        if positionPref1[0] < 0:
                                            positionPref1[0] = 0
                                            for i in range(len(positionPref1)-1):
                                                if positionPref1[i+1]-positionPref1[i] < oe['minDist'] + eps:
                                                    positionPref1[i+1]=positionPref1[i] + oe['minDist']
                                                else:
                                                    break
                                            if positionPref1[-1] > oe['length']:
                                                print('BIGERROR should not happen 2_2')
                                                print(positionPref1[-1])
                                                print(oe['length'])
                                if positionPref2 != []:
                                    if oe2['sign'] == 1:
                                        if positionPref2[-1] < 0:
                                            positionPref2[-1] = 0
                                            for i in range(-1,-1*(len(positionPref2)),-1):
                                                if positionPref2[i-1]-positionPref2[i] < oe2['minDist'] + eps:
                                                    positionPref2[i-1]=positionPref2[i] + oe2['minDist']
                                                else:
                                                    break
                                            if len(oe2['rRBC']) > 0:
                                                if positionPref2[0] > oe2['rRBC'][0]:
                                                    print('BIGERROR should not happen 3')
                                                    print(positionPref2[0])
                                                    print(oe2['rRBC'][0])
                                        if positionPref2[0] > oe2['length']:
                                            positionPref2[0] = oe2['length']
                                            for i in range(len(positionPref2)-1):
                                                if positionPref2[i]-positionPref2[i+1] < oe2['minDist'] + eps:
                                                    positionPref2[i+1]=positionPref2[i] - oe2['minDist']
                                                else:
                                                    break
                                            if positionPref2[-1] < 0:
                                                print('BIGERROR should not happen 3_2')
                                                print(positionPref2[-1])
                                    else:
                                        if positionPref2[-1] > oe2['length']:
                                            positionPref2[-1] = oe2['length']
                                            for i in range(-1,-1*(len(positionPref2)),-1):
                                                if positionPref2[i]-positionPref2[i-1] < oe2['minDist'] + eps:
                                                    positionPref2[i-1]=positionPref2[i] - oe2['minDist']
                                            if len(oe2['rRBC']) > 0:
                                                if positionPref2[0] < oe2['rRBC'][-1]:
                                                    print('BIGERROR should not happen 4')
                                                    print(positionPref2[0])
                                                    print(oe2['rRBC'][-1])
                                        if positionPref2[0] < 0:
                                            positionPref2[0] = 0
                                            for i in range(len(positionPref2)-1):
                                                if positionPref2[i+1]-positionPref2[i] < oe2['minDist'] + eps:
                                                    positionPref2[i+1]=positionPref2[i] + oe2['minDist']
                                                else:
                                                    break
                                            if positionPref2[-1] > oe2['length']:
                                                print('BIGERROR should not happen 4_2')
                                                print(positionPref2[-1])
                                                print(oe2['length'])

                            else:
                                #To begin with it is tried if all RBCs fit into the prefered outEdge. The time of arrival at the RBCs is taken into account
                                #RBCs which would be too close together are put into the other edge
                                #postion2/position3 is used if there is not enough space in the prefered outEdge and hence the RBC is moved to the other outEdge
                                #actual position of RBCs in the edges
                                positionPref2=[]
                                positionPref1=[]
                                inEPref1=[]
                                inEPref2=[]
                                indexPref1=[]
                                indexPref2=[]
                                #number of RBCs in the Edges
                                countPref1=0
                                countPref2=0
                                pref1Full = 0
                                pref2Full = 0
                                count1 = 0
                                count2 = 0
                                #Loop over all movable RBCs
                                for i in range(overshootsNo):
                                    index=-1*(i+1)
                                    index1=-1*(i+1) if oe['sign'] == 1.0 else i
                                    index2=-1*(i+1) if oe2['sign'] == 1.0 else i
                                    #posRBC1=position1[index1] if oe['sign'] == 1 else oe['length']-position1[index1]
                                    #posRBC2=position2[index2] if oe2['sign'] == 1 else oe2['length']-position2[index2]
                                    #check if RBC still fits into Prefered OutE
                                    if posNoBifEventsPref > countPref1 and pref1Full == 0:
                                        #Check if there are RBCs present in outEPref
                                        if positionPref1 != []:
                                            #Check if distance to preceding RBC is big enough
                                            dist1=positionPref1[-1]-position1[index1] if oe['sign'] == 1.0 \
                                                else position1[index1]-positionPref1[-1]
                                            #The distance is not big enough check if RBC fits into outEPref2
                                            if dist1 < oe['minDist']:
                                                #if RBCs are present in the outEdgePref2
                                                if posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                                    if positionPref2 != []:
                                                        dist2=positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                                            else position2[index2]-positionPref2[-1]
                                                        #Check if there is enough space in 2nd outEdge
                                                        #in case there is not enough space, check where the RBC is blocked the least amount of time
                                                        if dist2 < oe2['minDist']:
                                                            #Check if another RBCs still fits into the vessel
                                                            space1 =  positionPref1[-1] if oe['sign'] == 1.0 \
                                                                else oe['length']-positionPref1[-1]
                                                            if np.floor(space1/oe['minDist']) > 1:
                                                                timeBlocked1=(oe['minDist']-dist1)/oe['v']
                                                            else:
                                                                timeBlocked1=None
                                                                pref1Full=1
                                                            space2 =  positionPref2[-1] if oe2['sign'] == 1.0 \
                                                                else oe2['length']-positionPref2[-1]
                                                            if np.floor(space2/oe2['minDist']) > 1:
                                                                timeBlocked2=(oe2['minDist']-dist2)/oe2['v']
                                                            else:
                                                                timeBlocked2=None
                                                                pref2Full=1
                                                            if pref1Full == 1 and pref2Full == 1:
                                                                break
                                                            #Define newOutEdge
                                                            newOutEdge=0
                                                            if timeBlocked1 == None:
                                                                newOutEdge=2
                                                            elif timeBlocked2 == None:
                                                                newOutEdge=1
                                                            else:
                                                                if timeBlocked1 <= timeBlocked2:
                                                                    newOutEdge=1
                                                                else:
                                                                    newOutEdge=2
                                                            if newOutEdge == 1:
                                                                if oe['sign'] == 1.0:
                                                                    position1[index1]=positionPref1[-1]-oe['minDist']
                                                                    if position1[index1] > 0:
                                                                        positionPref1.append(position1[index1])
                                                                        inEPref1.append(inEdge[index])
                                                                        indexPref1.append(i)
                                                                        countPref1 += 1
                                                                        if inEdge[index] == 1:
                                                                            count1 += 1
                                                                        elif inEdge[index] == 2:
                                                                            count2 += 1
                                                                    else:
                                                                        print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 23')
                                                                        print(position1[index1])
                                                                        print(np.floor(space1/oe['minDist']))
                                                                        print(timeBlocked1)
                                                                        print(pref1Full)
                                                                else:
                                                                    position1[index1]=positionPref1[-1]+oe['minDist']
                                                                    if position1[index1] < oe['length']:
                                                                        positionPref1.append(position1[index1])
                                                                        inEPref1.append(inEdge[index])
                                                                        indexPref1.append(i)
                                                                        countPref1 += 1
                                                                        if inEdge[index] == 1:
                                                                            count1 += 1
                                                                        elif inEdge[index] == 2:
                                                                            count2 += 1
                                                                    else:
                                                                        print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 24')
                                                                        print(position1[index1])
                                                                        print(oe['length'])
                                                                        print(np.floor(space1/oe['minDist']))
                                                            elif newOutEdge == 2:
                                                                if oe2['sign'] == 1.0:
                                                                    position2[index2]=positionPref2[-1]-oe2['minDist']
                                                                    if position2[index2] > 0:
                                                                        positionPref2.append(position2[index2])
                                                                        inEPref2.append(inEdge[index])
                                                                        indexPref2.append(i)
                                                                        countPref2 += 1
                                                                        if inEdge[index] == 1:
                                                                            count1 += 1
                                                                        elif inEdge[index] == 2:
                                                                            count2 += 1
                                                                    else:
                                                                        print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 25')
                                                                        print(position2[index2])
                                                                        print(np.floor(space2/oe2['minDist']))
                                                                        print(timeBlocked2)
                                                                        print(pref2Full)
                                                                else:
                                                                    position2[index2]=positionPref2[-1]+oe2['minDist']
                                                                    if position2[index2] < oe2['length']:
                                                                        positionPref2.append(position2[index2])
                                                                        inEPref2.append(inEdge[index])
                                                                        indexPref2.append(i)
                                                                        countPref2 += 1
                                                                        if inEdge[index] == 1:
                                                                            count1 += 1
                                                                        elif inEdge[index] == 2:
                                                                            count2 += 1
                                                                    else:
                                                                        print('WARNING PROPAGATE RBC has been pushed outside SHOULD NOT HAPPEN 26')
                                                                        print(position2[index2])
                                                                        print(oe2['length'])
                                                                        print(np.floor(space2/oe2['minDist']))
                                                        #there is enough space for the RBC in outEPref2
                                                        else:
                                                            positionPref2.append(position2[index2])
                                                            inEPref2.append(inEdge[index])
                                                            indexPref2.append(i)
                                                            countPref2 += 1
                                                            if inEdge[index] == 1:
                                                                count1 += 1
                                                            elif inEdge[index] == 2:
                                                                count2 += 1
                                                    #there are no RBCs in outEdge2
                                                    else:
                                                        if oe2['sign'] == 1.0:
                                                            if len(oe2['rRBC'])>0:
                                                                if oe2['rRBC'][0]-position2[index2] < oe2['minDist']:
                                                                    position2[index2]=oe2['rRBC'][0]-oe2['minDist']
                                                            else:
                                                                if position2[index2] > oe2['length']:
                                                                    position2[index2]=oe2['length']
                                                        else:
                                                            if len(oe2['rRBC'])>0:
                                                                if position2[index2]-oe2['rRBC'][-1] < oe2['minDist']:
                                                                    position2[index2]=oe2['rRBC'][-1]+oe2['minDist']
                                                            else:
                                                                if position2[index2] < 0:
                                                                    position2[index2]=0
                                                        positionPref2.append(position2[index2])
                                                        inEPref2.append(inEdge[index])
                                                        indexPref2.append(i)
                                                        countPref2 += 1
                                                        if inEdge[index] == 1:
                                                            count1 += 1
                                                        elif inEdge[index] == 2:
                                                            count2 += 1
                                                #There is no space in outEPref2 --> RBC is pushed backwards in outEPref1
                                                else:
                                                    if oe['sign'] == 1.0:
                                                        position1[index1]=positionPref1[-1]-oe['minDist']
                                                        if position1[index1] > 0:
                                                            positionPref1.append(position1[index1])
                                                            inEPref1.append(inEdge[index])
                                                            indexPref1.append(i)
                                                            countPref1 += 1
                                                            if inEdge[index] == 1:
                                                                count1 += 1
                                                            elif inEdge[index] == 2:
                                                                count2 += 1
                                                        else:
                                                            pref1Full = 1
                                                            break
                                                    else:
                                                        position1[index1]=positionPref1[-1]+oe['minDist']
                                                        if position1[index1] < oe['length']:
                                                            positionPref1.append(position1[index1])
                                                            inEPref1.append(inEdge[index])
                                                            indexPref1.append(i)
                                                            countPref1 += 1
                                                            if inEdge[index] == 1:
                                                                count1 += 1
                                                            elif inEdge[index] == 2:
                                                                count2 += 1
                                                        else:
                                                            pref1Full = 1
                                                            break
                                            #If the RBC fits into outEPref1
                                            else:
                                                positionPref1.append(position1[index1])
                                                inEPref1.append(inEdge[index])
                                                indexPref1.append(i)
                                                countPref1 += 1
                                                if inEdge[index] == 1:
                                                    count1 += 1
                                                elif inEdge[index] == 2:
                                                    count2 += 1
                                        #There are not yet any new RBCs in outEdgePref
                                        else:
                                            if oe['sign'] == 1.0:
                                                if len(oe['rRBC'])>0:
                                                    if oe['rRBC'][0]-position1[index1] < oe['minDist']:
                                                        position1[index1]=oe['rRBC'][0]-oe['minDist']
                                                else:
                                                    if position1[index1] > oe['length']:
                                                        position1[index1]=oe['length']
                                            else:
                                                if len(oe['rRBC'])>0:
                                                    if position1[index1]-oe['rRBC'][-1] < oe['minDist']:
                                                        position1[index1]=oe['rRBC'][-1]+oe['minDist']
                                                else:
                                                    if position1[index1] < 0:
                                                        position1[index1]=0
                                            positionPref1.append(position1[index1])
                                            inEPref1.append(inEdge[index])
                                            indexPref1.append(i)
                                            countPref1 += 1
                                            if inEdge[index] == 1:
                                                count1 += 1
                                            elif inEdge[index] == 2:
                                                count2 += 1
                                    #The RBCs do not fit into the prefered outEdge anymore
                                    #Therefore they are either put in outEdge2 or outEdge3
                                    elif posNoBifEventsPref2 > countPref2 and pref2Full == 0:
                                        #Check if there are already new RBCs in outEPref2
                                        if positionPref2 != []:
                                            dist2=positionPref2[-1]-position2[index2] if oe2['sign'] == 1.0 \
                                                else position2[index2]-positionPref2[-1]
                                            if dist2 < oe2['minDist']:
                                                #RBCs are pushed backewards in Pref2
                                                if oe2['sign'] == 1.0:
                                                    position2[index2]=positionPref2[-1]-oe2['minDist']
                                                    if position2[index2] > 0:
                                                        positionPref2.append(position2[index2])
                                                        inEPref2.append(inEdge[index])
                                                        indexPref2.append(i)
                                                        countPref2 += 1
                                                        if inEdge[index] == 1:
                                                            count1 += 1
                                                        elif inEdge[index] == 2:
                                                            count2 += 1
                                                    else:
                                                        pref2Full = 1
                                                        break
                                                else:
                                                    position2[index2]=positionPref2[-1]+oe2['minDist']
                                                    if position2[index2] < oe2['length']:
                                                        positionPref2.append(position2[index2])
                                                        inEPref2.append(inEdge[index])
                                                        indexPref2.append(i)
                                                        countPref2 += 1
                                                        if inEdge[index] == 1:
                                                            count1 += 1
                                                        elif inEdge[index] == 2:
                                                            count2 += 1
                                                    else:
                                                        pref2Full = 1
                                                        break
                                            #There is enough space for the RBCs in the outEdge 2
                                            else:
                                                positionPref2.append(position2[index2])
                                                inEPref2.append(inEdge[index])
                                                indexPref2.append(i)
                                                countPref2 += 1
                                                if inEdge[index] == 1:
                                                    count1 += 1
                                                elif inEdge[index] == 2:
                                                    count2 += 1
                                        #There are not yet any RBCs in outEdge 2
                                        else:
                                            if oe2['sign'] == 1.0:
                                                if len(oe2['rRBC']) > 0:
                                                    if oe2['rRBC'][0]-position2[index2] < oe2['minDist']:
                                                        position2[index2]=oe2['rRBC'][0]-oe2['minDist']
                                                else:
                                                    if position2[index2] > oe2['length']:
                                                        position2[index2]=oe2['length']
                                            else:
                                                if len(oe2['rRBC']) > 0:
                                                    if position2[index2]-oe2['rRBC'][-1] < oe2['minDist']:
                                                        position2[index2]=oe2['rRBC'][-1]+oe2['minDist']
                                                else:
                                                    if position2[index2] < 0:
                                                        position2[index2]=0
                                            positionPref2.append(position2[index2])
                                            inEPref2.append(inEdge[index])
                                            indexPref2.append(i)
                                            countPref2 += 1
                                            if inEdge[index] == 1:
                                                count1 += 1
                                            elif inEdge[index] == 2:
                                                count2 += 1
                                    #There is no more space for further RBCs
                                    else:
                                        break
                                if countPref2+countPref1 != overshootsNo:
                                    overshootsNo = countPref2+countPref1
                            #Add rbcs to outEPref
                            nRBCint=oe['nRBCint']
                            for i in positionPref1:
                                nRBCint += self._nRBC_mean_inflow(i,oe,dt)
                            oe['nRBCint']=nRBCint
                            if oe['sign'] == 1.0:
                                oe['rRBC']=np.concatenate([positionPref1[::-1], oe['rRBC']])
                            else:
                                oe['rRBC']=np.concatenate([oe['rRBC'],positionPref1])
                            #Add rbcs to outEPref2       
                            for i in positionPref2:
                                nRBCint += self._nRBC_mean_inflow(i,oe2,dt)
                            oe2['nRBCint']=nRBCint
                            if oe2['sign'] == 1.0:
                                oe2['rRBC']=np.concatenate([positionPref2[::-1], oe2['rRBC']])
                            else:
                                oe2['rRBC']=np.concatenate([oe2['rRBC'],positionPref2])
                            #Remove RBCs from old Edge 1
                            if count1 > 0:
                                nRBCint=e['nRBCint']
                                if e['sign'] == 1:
                                    for i in range(count1):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold[-1*(i+1)],e,dt)
                                else:
                                    for i in range(count1):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold[i],e,dt)
                                if sign == 1.0:
                                    e['rRBC']=e['rRBC'][:-count1]
                                else:
                                    e['rRBC']=e['rRBC'][count1::]
                                nRBCint += len(e['nRBC'])
                                e['nRBCint']=nRBCint
                            if noBifEvents2 > 0 and count2 > 0:
                                nRBCint=e2['nRBCint']
                                if e2['sign'] == 1:
                                    for i in range(count2):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold2[-1*(i+1)],e2,dt)
                                else:
                                    for i in range(count2):
                                        nRBCint += self._nRBC_mean_outflow(rRBCold2[i],e2,dt)
                                #Remove RBCs from old Edge 2
                                if sign2 == 1.0:
                                    e2['rRBC']=e2['rRBC'][:-count2]
                                else:
                                    e2['rRBC']=e2['rRBC'][count2::]
                                nRBCint += len(e2['rRBC'])
                                e2['nRBCint']=nRBCint
                        #OutEdges are currently blocked, no bifurcation events possible
                        else:
                            countPref1=0
                            countPref2=0
                            count1=0
                            count2=0
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        #InEdge 1
                        noStuckRBCs1=len(bifRBCsIndex1)-count1
                        for i in range(noStuckRBCs1):
                            index=-1*(i+1) if sign == 1.0 else i
                            e['rRBC'][index]=e['length']-i*e['minDist'] if sign == 1.0 else 0+i*e['minDist']
                        #Recheck if the distance between the newly introduces RBCs is still big enough 
                        if len(e['rRBC']) >1.0:
                            moved = 0
                            if sign == 1.0:
                                count = 0
                                for i in range(-1,-1*(len(e['rRBC'])),-1):
                                    index=i-1
                                    if e['rRBC'][i] < e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]-e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs1 and moved == 0:
                                        break
                            else:
                                count = 0
                                for i in range(len(e['rRBC'])-1):
                                    index=i+1
                                    if e['rRBC'][i] > e['rRBC'][index] or abs(e['rRBC'][i]-e['rRBC'][index]) < e['minDist']:
                                        e['rRBC'][index]=e['rRBC'][i]+e['minDist']
                                        moved = 1
                                    else:
                                        moved = 0
                                    count += 1
                                    if count >= noStuckRBCs1+1 and moved == 0:
                                        break
                        #Deal with RBCs which could not be reassigned to the new edge because of a traffic jam
                        #InEdge 2
                        if noBifEvents2 > 0:
                            noStuckRBCs2=len(bifRBCsIndex2)-count2
                            for i in range(noStuckRBCs2):
                                index=[-1*(i+1) if sign2 == 1.0 else i]
                                e2['rRBC'][index]=[e2['length']-i*e2['minDist'] if sign2 == 1.0 else 0+i*e2['minDist']]
                            #Recheck if the distance between the newly introduces RBCs is still big enough 
                            if len(e2['rRBC']) >1:
                                moved = 0
                                if sign2 == 1.0:
                                    count = 0
                                    for i in range(-1,-1*(len(e2['rRBC'])),-1):
                                        index=i-1
                                        if e2['rRBC'][i] < e2['rRBC'][index] or abs(e2['rRBC'][i]-e2['rRBC'][index]) < e2['minDist']:
                                            e2['rRBC'][index]=e2['rRBC'][i]-e2['minDist']
                                            moved = 1
                                        else:
                                            moved = 0
                                        count += 1
                                        if count >= noStuckRBCs2 and moved == 0:
                                            break
                                else:
                                    count = 0
                                    for i in range(len(e2['rRBC'])-1):
                                        index=i+1
                                        if e2['rRBC'][i] > e2['rRBC'][index] or abs(e2['rRBC'][i]-e2['rRBC'][index]) < e2['minDist']:
                                            e2['rRBC'][index]=e2['rRBC'][i]+e2['minDist']
                                            moved = 1
                                        else:
                                            moved = 0
                                        count += 1
                                        if count >= noStuckRBCs2+1 and moved == 0:
                                            break
            #-------------------------------------------------------------------------------------------
            rRBC = []
            rRBC2 = []
            rRBC3 = []
            if e['httBC'] is not None:
                boolHttEdge = 1
                rRBC = []
                lrbc = e['minDist']
                htt = e['httBC']
                length = e['length']
                nMaxNew=e['nMax']-len(e['rRBC'])
                if len(e['rRBC']) > 0:
                    #if cum_length > distToFirst:
                    posFirst=e['rRBC'][0] if e['sign']==1.0 else e['length']-e['rRBC'][-1]
                    e['posFirst_last']=posFirst
                    e['v_last'] = e['v']
                    cum_length = posFirst
                else:
                    cum_length = e['posFirst_last'] + e['v_last'] * dt
                    posFirst = cum_length
                    e['posFirst_last']=posFirst
                    if e['v'] > e['v_last']:
                        e['v_last']=e['v']
                while cum_length >= lrbc and nMaxNew > 0:
                    if len(e['keep_rbcs']) != 0:
                        if posFirst - e['keep_rbcs'][0] >= 0:
                            if posFirst - e['keep_rbcs'][0] > e['length']:
                                rRBC.append(e['length'])
                                posFirst=e['length']
                            else:
                                rRBC.append(posFirst - e['keep_rbcs'][0])
                                posFirst=posFirst - e['keep_rbcs'][0]
                            nMaxNew += -1
                            cum_length = posFirst
                            e['keep_rbcs']=[]
                            e['posFirst_last']=posFirst
                            #e['v_last']=e['v']
                        else:
                            break
                    else:
                        #number of RBCs randomly chosen to average htt
                        stdout.flush()
                        number=np.exp(e['logNormal'][0]+e['logNormal'][1]*np.random.randn(1)[0])
                        #self._spacing.append(number)
                        spacing = lrbc+lrbc*number
                        if posFirst - spacing >= 0:
                            if posFirst - spacing > e['length']:
                                rRBC.append(e['length'])
                                posFirst=e['length']
                            else:
                                rRBC.append(posFirst - spacing)
                                posFirst=posFirst - spacing
                            nMaxNew += -1
                            cum_length = posFirst
                            e['posFirst_last']=posFirst
                            #e['v_last']=e['v']
                        else:
                            e['keep_rbcs']=[spacing]
                            #e['v_last']=e['v']
                            if len(rRBC) == 0:
                                e['posFirst_last']=posFirst
                            else:
                                e['posFirst_last']=rRBC[-1]
                            break
                if len(e['keep_rbcs']) == 0:
                    number=np.exp(e['logNormal'][0]+e['logNormal'][1]*np.random.randn(1)[0])
                    spacing = lrbc+lrbc*number
                    e['keep_rbcs']=[spacing]
                rRBC = np.array(rRBC)
                if len(rRBC) >= 1.:
                    if e['sign'] == 1:
                        e['rRBC'] = np.concatenate([rRBC[::-1], e['rRBC']])
                        vertexUpdate.append(e.target)
                        vertexUpdate.append(e.source)
                        edgeUpdate.append(ei)
                    else:
                        rRBC=length-rRBC
                        e['rRBC'] = np.concatenate([e['rRBC'], rRBC])
                        vertexUpdate.append(e.target)
                        vertexUpdate.append(e.source)
                        edgeUpdate.append(ei)
                    nRBCint=e['nRBCint']
                    for i in rRBC:
                        nRBCint += self._nRBC_mean_inflow(i,e,dt)
                    e['nRBCint']=nRBCint
            if noBifEvents > 0:
                if G.vs[vi]['vType'] == 6 or G.vs[vi]['vType'] == 4:
                    #Check if httBC exists
                    boolHttEdge2 = 0
                    if e2['httBC'] is not None:
                        boolHttEdge2 = 1
                        rRBC2 = []
                        lrbc = e2['minDist']
                        htt = e2['httBC']
                        length = e2['length']
                        nMaxNew=e2['nMax']-len(e2['rRBC'])
                        if len(e2['rRBC']) > 0:
                            posFirst=e2['rRBC'][0] if e2['sign']==1.0 else e2['length']-e2['rRBC'][-1]
                            e2['posFirst_last']=posFirst
                            e2['v_last'] = e2['v']
                            cum_length = posFirst
                        else:
                            cum_length = e2['posFirst_last'] + e2['v_last'] * dt
                            posFirst = cum_length
                            e2['posFirst_last']=posFirst
                            if e2['v'] > e2['v_last']:
                                e2['v_last']=e2['v']
                        while cum_length >= lrbc and nMaxNew > 0:
                            if len(e2['keep_rbcs']) != 0:
                                if posFirst - e2['keep_rbcs'][0] >= 0:
                                    if posFirst - e2['keep_rbcs'][0] > e2['length']:
                                        rRBC2.append(e2['length'])
                                        posFirst=e2['length']
                                    else:
                                        rRBC2.append(posFirst - e2['keep_rbcs'][0])
                                        posFirst=posFirst - e2['keep_rbcs'][0]
                                    nMaxNew += -1
                                    cum_length = posFirst
                                    e2['keep_rbcs']=[]
                                    e2['posFirst_last']=posFirst
                                    #e2['v_last']=e2['v']
                                else:
                                    if len(e2['rRBC']) > 0:
                                        e2['posFirst_last'] = posFirst
                                        #e2['v_last']=e2['v']
                                    break
                            else:
                                #number of RBCs randomly chosen to average htt
                                number=np.exp(e2['logNormal'][0]+e2['logNormal'][1]*np.random.randn(1)[0])
                                spacing = lrbc+lrbc*number
                                if posFirst - spacing >= 0:
                                    if posFirst - spacing > e2['length']:
                                        rRBC2.append(e2['length'])
                                        posFirst=e2['length']
                                    else:
                                        rRBC2.append(posFirst - spacing)
                                        posFirst=posFirst - spacing
                                    nMaxNew += -1
                                    cum_length = posFirst
                                    e2['posFirst_last']=posFirst
                                    #e2['v_last']=e2['v']
                                else:
                                    e2['keep_rbcs']=[spacing]
                                    #e2['v_last']=e2['v']
                                    if len(rRBC2) == 0:
                                        e2['posFirst_last']=posFirst
                                    else:
                                        e2['posFirst_last']=rRBC2[-1]
                                    break
                        if len(e2['keep_rbcs']) == 0:
                            number=np.exp(e2['logNormal'][0]+e2['logNormal'][1]*np.random.randn(1)[0])
                            spacing = lrbc+lrbc*number
                            e2['keep_rbcs']=[spacing]
                        rRBC2 = np.array(rRBC2)
                        if len(rRBC2) >= 1.:
                            if e2['sign'] == 1:
                                e2['rRBC'] = np.concatenate([rRBC2[::-1], e2['rRBC']])
                                vertexUpdate.append(e2.target)
                                vertexUpdate.append(e2.source)
                                edgeUpdate.append(e2.index)
                            else:
                                rRBC2=length-rRBC2
                                e2['rRBC'] = np.concatenate([e2['rRBC'], rRBC2])
                                vertexUpdate.append(e2.target)
                                vertexUpdate.append(e2.source)
                                edgeUpdate.append(e2.index)
                            nRBCint=e2['nRBCint']
                            for i in rRBC2:
                                nRBCint += self._nRBC_mean_inflow(i,e2,dt)
                            e2['nRBCint']=nRBCint
                    if G.vs[vi]['vType']==4 and len(inflowEdges) > 2:
                        #Check if httBC exists
                        boolHttEdge3 = 0
                        if e3['httBC'] is not None:
                            boolHttEdge3 = 1
                            rRBC3 = []
                            lrbc = e3['minDist']
                            htt = e3['httBC']
                            length = e3['length']
                            nMaxNew=e3['nMax']-len(e3['rRBC'])
                            if len(e3['rRBC']) > 0:
                                posFirst=e3['rRBC'][0] if e3['sign']==1.0 else e3['length']-e3['rRBC'][-1]
                                e3['posFirst_last']=posFirst
                                e3['v_last'] = e3['v']
                                cum_length = posFirst
                            else:
                                cum_length = e3['posFirst_last'] + e3['v_last'] * dt
                                posFirst = cum_length
                                e3['posFirst_last']=posFirst
                                if e3['v'] > e3['v_last']:
                                    e3['v_last']=e3['v']
                            while cum_length >= lrbc and nMaxNew > 0:
                                if len(e3['keep_rbcs']) != 0:
                                    if posFirst - e3['keep_rbcs'][0] >= 0:
                                        if posFirst - e3['keep_rbcs'][0] > e3['length']:
                                            rRBC3.append(e3['length'])
                                            posFirst=e3['length']
                                        else:
                                            rRBC3.append(posFirst - e3['keep_rbcs'][0])
                                            posFirst=posFirst - e3['keep_rbcs'][0]
                                        nMaxNew += -1
                                        cum_length = posFirst
                                        e3['keep_rbcs']=[]
                                        e3['posFirst_last']=posFirst
                                        #e3['v_last']=e3['v']
                                    else:
                                        if len(e3['rRBC']) > 0:
                                            e3['posFirst_last'] = posFirst
                                            #e3['v_last']=e3['v']
                                        break
                                else:
                                    #number of RBCs randomly chosen to average htt
                                    number=np.exp(e3['logNormal'][0]+e3['logNormal'][1]*np.random.randn(1)[0])
                                    spacing = lrbc+lrbc*number
                                    if posFirst - spacing >= 0:
                                        if posFirst - spacing > e3['length']:
                                            rRBC3.append(e3['length'])
                                            posFirst=e3['length']
                                        else:
                                            rRBC3.append(posFirst - spacing)
                                            posFirst=posFirst - spacing
                                        nMaxNew += -1
                                        cum_length = posFirst
                                        e3['posFirst_last']=posFirst
                                        #e3['v_last']=e3['v']
                                    else:
                                        e3['keep_rbcs']=[spacing]
                                        #e3['v_last']=e3['v']
                                        if len(rRBC3) == 0:
                                            e3['posFirst_last']=posFirst
                                        else:
                                            e3['posFirst_last']=rRBC3[-1]
                                        break
                            if len(e3['keep_rbcs']) == 0:
                                number=np.exp(e3['logNormal'][0]+e3['logNormal'][1]*np.random.randn(1)[0])
                                spacing = lrbc+lrbc*number
                                e3['keep_rbcs']=[spacing]
                            rRBC3 = np.array(rRBC3)
                            if len(rRBC3) >= 1.:
                                if e3['sign'] == 1:
                                    e3['rRBC'] = np.concatenate([rRBC3[::-1], e3['rRBC']])
                                    vertexUpdate.append(e3.target)
                                    vertexUpdate.append(e3.source)
                                    edgeUpdate.append(e3.index)
                                else:
                                    rRBC3=length-rRBC3
                                    e3['rRBC'] = np.concatenate([e3['rRBC'], rRBC3])
                                    vertexUpdate.append(e3.target)
                                    vertexUpdate.append(e3.source)
                                    edgeUpdate.append(e3.index)
                                nRBCint=e3['nRBCint']
                                for i in rRBC3:
                                    nRBCint += self._nRBC_mean_inflow(i,e3,dt)
                                e3['nRBCint']=nRBCint
            if noBifEvents != 0 or boolHttEdge == 1 or boolHttEdge2==1 or boolHttEdge3==1:
                nRBCSumAfter=0
                for i in edgesInvolved:
                    G.es[i]['nRBC']=len(G.es[i]['rRBC'])
                    nRBCSumAfter += G.es[i]['nRBC']
                if nRBCSumBefore != nRBCSumAfter:
                    #Check if outflow
                    if G.vs[vi]['vType'] == 2:
                        if nRBCSumAfter + noBifEvents != nRBCSumBefore:
                            print('BIGERROR RBC CONSERVATION at outlet')
                            print(nRBCSumAfter)
                            print(nRBCSumBefore)
                            print(noBifEvents)
                            print(vi)
                            print(G.vs[vi]['av'])
                            print(G.vs[vi]['vv'])
                            print(G.adjacent(vi))
                            print(G.es[G.adjacent(vi)]['httBC'])
                            print(edgesInvolved)
                            print('boolHtt')
                            print(boolHttEdge)
                            print(boolHttEdge2)
                            print(boolHttEdge3)
                    else:
                        if boolHttEdge == 1 or boolHttEdge2==1 or boolHttEdge3==1:
                            rbcsAdded=0
                            if boolHttEdge == 1:
                                rbcsAdded += len(rRBC)
                            if boolHttEdge2 == 1:
                                rbcsAdded += len(rRBC2)
                            if boolHttEdge3 == 1:
                                rbcsAdded += len(rRBC3)
                            if nRBCSumAfter - rbcsAdded != nRBCSumBefore:
                                    print('BIGERROR RBC CONSERVATION at inlet')
                                    print(nRBCSumAfter)
                                    print(nRBCSumBefore)
                                    print('boolHtt')
                                    print(boolHttEdge)
                                    print(boolHttEdge2)
                                    print(boolHttEdge3)
                                    print(len(rRBC))
                                    print(len(rRBC2))
                                    print(len(rRBC3))
                                    print(noBifEvents)
                                    print(vi)
                                    print(G.vs[vi]['vType'])
                                    print(edgesInvolved)
                        else:
                            print('BIGERROR RBC CONSERVATION somewhere')
                            print(G.vs['vType'][vi])
                            print(G.vs['inflowE'][vi])
                            print(G.vs['outflowE'][vi])
                            print(G.es[G.vs['inflowE'][vi]]['sign'])
                            print(G.es[G.vs['outflowE'][vi]]['sign'])
                for i in edgesInvolved:
                    edge = G.es[i]
                    if len(edge['rRBC']) > 0:
                        if edge['rRBC'][0] > edge['rRBC'][-1] + eps:
                            print('BIGERROR BEGINNING END')
                            print(i)
                            print(edge['rRBC'][0])
                            print(edge['rRBC'][-1])
                            print(G.vs[vi]['vType'])
                        if edge['rRBC'][0] < 0 - eps or edge['rRBC'][-1] > edge['length'] + eps:
                            print('BIGERROR BEGINNING END 2')
                            print(i)
                            print('vi')
                            print(vi)
                            print(edge['rRBC'][0])
                            print(edge['rRBC'][-1])
                            print(edge['rRBC'])
                            print(edge['length'])
                            print(edge['sign'])
                            print(G.vs[vi]['vType'])
                            print(G.vs[vi]['inflowE'])
                            print(G.vs[vi]['outflowE'])
                            print(boolHttEdge)
                            print(boolHttEdge2)
                            print(boolHttEdge3)
                    for j in range(len(edge['rRBC'])-1):
                        if edge['rRBC'][j] > edge['rRBC'][j+1] or edge['rRBC'][j+1]-edge['rRBC'][j] < edge['minDist']-100*eps:
                            print('BIGERROR BEGINNING END 3')
                            print(i)
                            print(edge['rRBC'][j])
                            print(edge['rRBC'][j+1])
                            print(edge['rRBC'][j+1]-edge['rRBC'][j])
                            print(edge['minDist'])
                            print(edge['sign'])
                            print(G.vs[vi]['vType'])
            if overshootsNo != 0:
                if self._capDil != 0:
                    if vi in upstreamDivBifs:
                        print(vi)
                        vIndex = upstreamDivBifs.index(vi)
                        print(vIndex)
                        print(divBifEdges[vIndex])
                        print(G.vs[vi]['outflowE'])
                        print(G.adjacent(vi))
                        upstreamDivBifsCount[vIndex] +=1
                        if G.vs[vi]['vType']==3:
                            #WARNING this my lead to errors if the number of divergent bifurcations changes during simulation
                            index2=divBifEdges[vIndex].index(oe.index)
                            analyzeDivBifEvents[vIndex][index2] += len(positionPref1)
                            index2=divBifEdges[vIndex].index(oe2.index)
                            analyzeDivBifEvents[vIndex][index2] += len(positionPref2)
                            if len(divBifEdges[vIndex]) > 2:
                                index2=divBifEdges[vIndex].index(oe3.index)
                                analyzeDivBifEvents[vIndex][index2] += len(positionPref3)
                        else:
                            upstreamDivBifsNotDiv[vIndex] += 1
                vertexUpdate.append(e.target)
                vertexUpdate.append(e.source)
                for i in edgesInvolved:
                    edgeUpdate.append(i)
                if boolHttEdge:
                    for i in e['rRBC']:
                        if i < 0 or i > e['length']:
                            print('BIGERROREND')
                            print('LOOK2 HTTBC Edge')
                            print(e['rRBC'][0])
                            print(e['rRBC'][-1])
                            print(e['length'])
                            print(vi)
                            print(e.tuple)
                            print(noBifEvents)
                if boolHttEdge2:
                    for i in e2['rRBC']:
                        if i < 0 or i > e2['length']:
                            print('BIGERROREND')
                            print('LOOK2 HTTBC Edge 2')
                            print(e2['rRBC'][0])
                            print(e2['rRBC'][-1])
                            print(e2['length'])
                            print(e2['sign'])
                            print(vi)
                            print(e2.tuple)
                            print('total number of bif events')
                            print(noBifEvents)
                            print(noBifEvents1)
                            print(noBifEvents2)
                            print('bifurcation type')
                            print(G.vs[vi]['vType'])
                if boolHttEdge3:
                    for i in e3['rRBC']:
                        if i < 0 or i > e3['length']:
                            print('BIGERROREND')
                            print('LOOK2 HTTBC Edge 3')
                            print(e3['rRBC'][0])
                            print(e3['rRBC'][-1])
                            print(e3['length'])
                            print(vi)
                            print(e3.tuple)
                            print(noBifEvents)
                if self._analyzeBifEvents:
                    if G.vs['vType'][vi] == 3 or G.vs['vType'][vi] == 5:
                        rbcMoved += overshootsNo
                    elif G.vs['vType'][vi] == 6:
                        rbcMoved += count1 + count2
                    elif G.vs['vType'][vi] == 4:
                        rbcMoved += count1 + count2
                        if len(inflowEdges) > 2:
                            if count3 > 0:
                                rbcMoved += count3
                if self._analyzeBifEvents:
                    if G.vs['vType'][vi] == 3 or G.vs['vType'][vi] == 5:
                        rbcsMovedPerEdge.append(overshootsNo)
                        edgesWithMovedRBCs.append(e.index)
                    elif G.vs['vType'][vi] == 6:
                        if count1 > 0:
                            rbcsMovedPerEdge.append(count1)
                            edgesWithMovedRBCs.append(e.index)
                        if count2 > 0:
                            edgesWithMovedRBCs.append(e2.index)
                            rbcsMovedPerEdge.append(count2)
                    elif G.vs['vType'][vi] == 4:
                        if count1 > 0:
                            rbcsMovedPerEdge.append(count1)
                            edgesWithMovedRBCs.append(e.index)
                        if count2 > 0:
                            edgesWithMovedRBCs.append(e2.index)
                            rbcsMovedPerEdge.append(count2)
                        if len(inflowEdges) > 2:
                            if count3 > 0:
                                rbcsMovedPerEdge.append(count3)
                                edgesWithMovedRBCs.append(e.index)

        #-------------------------------------------------------------------------------------------
        self._vertexUpdate=np.unique(vertexUpdate)
        edgeUpdate=np.unique(edgeUpdate)
        self._edgeUpdate=edgeUpdate.tolist()
        if self._capDil != 0:
            self._upstreamDivBifs = upstreamDivBifs
            self._DivBifEdges=divBifEdges
            self._analyzeDivBifEvents=analyzeDivBifEvents
            self._upstreamDivBifsCount=upstreamDivBifsCount
            self._upstreamDivBifsNotDiv=upstreamDivBifsNotDiv
        G.es['nRBC'] = [len(e['rRBC']) for e in G.es]
        if self._analyzeBifEvents:
            self._rbcsMovedPerEdge.append(rbcsMovedPerEdge)
            self._edgesWithMovedRBCs.append(edgesWithMovedRBCs)
            self._rbcMoveAll.append(rbcMoved)
        self._G=G
    #--------------------------------------------------------------------------
    #@profile
    def evolve(self, time, method, dtfix,**kwargs):
        """Solves the linear system A x = b using a direct or AMG solver.
        INPUT: time: The duration for which the flow should be evolved. In case of
                      Reset in plotPrms or samplePrms = False, time is the duration 
                      which is added
               method: Solution-method for solving the linear system. This can
                       be either 'direct' or 'iterative'
               dtfix: given timestep
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
         OUTPUT: None (files are written to disk)
        """
        G=self._G
        tPlot = self._tPlot # deepcopy, since type is float
        tSample = self._tSample # deepcopy, since type is float
        filenamelist = self._filenamelist
        self._dt=dtfix
        timelist = self._timelist
        #filenamelistAvg = self._filenamelistAvg
        timelistAvg = self._timelistAvg

        if 'init' in kwargs.keys():
            init=kwargs['init']
        else:
            init=self._init

        SampleDetailed=False
        if 'SampleDetailed' in kwargs.keys():
            SampleDetailed=kwargs['SampleDetailed']

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
                self._tSample = 0.0
                self._sampledict = {}
                #self._transitTimeDict = {}
                #filenamelistAvg = []
                timelistAvg = []
            else:
                if 'iterFinalSample' not in G.attributes():
                    G['iterFinalSample'] = 0
                self._tSample = G['iterFinalSample']
                sStart = G['iterFinalSample']+sStart+sStep
                sStop = G['iterFinalSample']+sStop

        t1 = ttime.time()
        if init:
            self._t = 0.0
            BackUpTStart=0.1*time
            #BackUpTStart=time/30
            BackUpT=0.1*time
            #BackUpT=time/30.
            BackUpCounter=0
        else:
            if 'dtFinal' not in G.attributes():
                G['dtFinal'] = 0
            if 'BackUpCounter' not in G.attributes():
                G['BackUpCounter'] = 0
            if 'iterFinalSample' not in G.attributes():
                G['iterFinalSample'] = 0
            self._t = G['dtFinal']
            self._tSample=G['iterFinalSample']
            BackUpT=0.1*time
            #BackUpT=time/65.
            #BackUpT=0.05*time
            #print('Simulation starts at')
            #print(self._t)
            #print('First BackUp should be done at')
            time = G['dtFinal']+time
            BackUpCounter=G['BackUpCounter']+1
            BackUpTStart=G['dtFinal']+BackUpT
            #print(BackUpTStart)
            #print('BackUp should be done every')
            #print(BackUpT)

        #Convert 'pBC' ['mmHG'] to default Units
        for v in G.vs:
            if v['pBC'] != None:
                v['pBC']=v['pBC']*self._scaleToDef

        tSample = self._tSample
        start_timeTot=ttime.time()
        t=self._t
        iteration=0
        self._edgeUpdate=[]
        #htts0=[]
        #keepRBCs0=[]
        #numbers0=[]
        #vf0=[]
        #v0=[]
        while True:
            if t >= time:
                break
            #htts0.append(G.es[0]['htt'])
            #keepRBCs0.append(G.es[0]['keep_rbcs'][0])
            #numbers0.append((G.es[0]['keep_rbcs'][0]-G.es[0]['minDist'])/G.es[0]['minDist'])
            #vf0.append(G.es[0]['v']/(G.es[0]['flow']/(0.25*np.pi*G.es[0]['diameter']**2)))
            #v0.append(G.es[0]['v'])
            iteration += 1
            start_time=ttime.time()
            self._update_eff_resistance_and_LS(None, self._vertexUpdate)
            #print('Matrix updated')
            self._solve(method, **kwargs)
            #print('Matrix solved')
            stdout.flush()
            self._G.vs['pressure'] = deepcopy(self._x)
            #print('Pressure copied')
            self._update_flow_and_velocity()
            #print('Flow updated')
            self._update_flow_sign()
            #print('Flow sign updated')
            self._verify_mass_balance()
            #print('Mass balance verified updated')
            self._update_out_and_inflows_for_vertices()
            #print('In and outflows updated')
            #TODO plotting
            #if doPlotting and tPlot >= pStart and tPlot <= pStop:
            #    filename = 'iter_'+str(int(round(tPlot)))+'.vtp'
                #filename = 'iter_'+('%.3f' % t)+'.vtp'
            #    filenamelist.append(filename)
            #    timelist.append(tPlot)
            #    self._plot_rbc(filename)
            #    pStart = tPlot + pStep
                #self._sample()
                #filename = 'sample_'+str(int(round(tPlot)))+'.vtp'
                #self._sample_average()
            if SampleDetailed:
                #print('sample detailed')
                stdout.flush()
                self._t=t
                self._tSample=tSample
                self._sample()
                filenameDetailed ='G_iteration_'+str(iteration)+'.pkl'
                #Convert deaultUnits to ['mmHG']
                #for 'pBC' and 'pressure'
                for v in G.vs:
                    if v['pBC'] != None:
                        v['pBC']=v['pBC']/self._scaleToDef
                    v['pressure']=v['pressure']/self._scaleToDef
                vgm.write_pkl(G,filenameDetailed)
                #Convert 'pBC' ['mmHG'] to default Units
                for v in G.vs:
                    if v['pBC'] != None:
                        v['pBC']=v['pBC']*self._scaleToDef
                    v['pressure']=v['pressure']*self._scaleToDef
            else:
                if doSampling and tSample >= sStart and tSample <= sStop:
                    #print('DO sampling')
                    stdout.flush()
                    self._t=t
                    self._tSample=tSample
                    #print('start sampling')
                    stdout.flush()
                    self._sample()
                    sStart = tSample + sStep
                    #print('sampling DONE')
                    if t > BackUpTStart:
                        #print('BackUp should be done')
                        #print(BackUpCounter)
                        stdout.flush()
                        G['dtFinal']=t
                        G['iterFinalSample']=tSample
                        G['BackUpCounter']=BackUpCounter
                        if self._analyzeBifEvents:
                            G['rbcsMovedPerEdge']=self._rbcsMovedPerEdge
                            G['edgesMovedRBCs']=self._edgesWithMovedRBCs
                            G['rbcMovedAll']=self._rbcMoveAll
                        #G['spacing']=self._spacing
                        filename1='sampledict_BackUp_'+str(BackUpCounter)+'.pkl'
                        filename2='G_BackUp'+str(BackUpCounter)+'.pkl'
                        if self._capDil != 0:
                            analyzeDivBifs={}
                            analyzeDivBifs['upstreamDivBifs']=self._upstreamDivBifs
                            analyzeDivBifs['divBifEdges']=self._DivBifEdges
                            analyzeDivBifs['analyzeDivBifEvents']=self._analyzeDivBifEvents
                            analyzeDivBifs['upstreamDivBifsCount']=self._upstreamDivBifsCount
                            analyzeDivBifs['upstreamDivBifsNotDiv']=self._upstreamDivBifsNotDiv
                            vgm.write_pkl(analyzeDivBifs,'analyzeDivBifs_BackUp'+str(BackUpCounter)+'.pkl')
                        self._sample_average()
                        #print(filename1)
                        #print(filename2)
                        #Convert deaultUnits to 'pBC' ['mmHG']
                        for v in G.vs:
                            if v['pBC'] != None:
                                v['pBC']=v['pBC']/self._scaleToDef
                            v['pressure']=v['pressure']/self._scaleToDef
                        g_output.write_pkl(self._sampledict,filename1)
                        vgm.write_pkl(G,filename2)
                        self._sampledict = {}
                        self._sampledict['averagedCount']=G['averagedCount']
                        #Convert 'pBC' ['mmHG'] to default Units
                        for v in G.vs:
                            if v['pBC'] != None:
                                v['pBC']=v['pBC']*self._scaleToDef
                            v['pressure']=v['pressure']*self._scaleToDef
                        BackUpCounter += 1
                        BackUpTStart += BackUpT
                        #print('BackUp Done')
            #print('START RBC propagate')
            stdout.flush()
            edgeUpdateOld = deepcopy(self._edgeUpdate)
            self._propagate_rbc()
            #print('RBCs propagated')
            edgeUpdate=edgeUpdateOld + self._edgeUpdate
            edgeUpdate=np.unique(edgeUpdate)
            edgeUpdate=edgeUpdate.tolist()
            self._update_hematocrit(edgeUpdate)
            #print('Hematocrit updated')
            stdout.flush()
            tPlot = tPlot + self._dt
            self._tPlot = tPlot
            tSample = tSample + self._dt
            self._tSample = tSample
            t = t + self._dt
            log.info(t)
            #print('TIME')
            #print(t)
            #print("Execution Time Loop:")
            #print(ttime.time()-start_time, "seconds")
            #print(' ')
            #print(' ')
            #stdout.write("\r%f" % tPlot)
            #stdout.flush()
        #stdout.write("\rDone. t=%f        \n" % tPlot)
        #log.info("Time taken: %.2f" % (ttime.time()-t1))
        #print("Execution Time:")
        #print(ttime.time()-start_timeTot, "seconds")

        self._update_eff_resistance_and_LS(None, None)
        self._solve(method, **kwargs)
        self._G.vs['pressure'] = deepcopy(self._x)
        #print('Pressure copied')
        self._update_flow_and_velocity()
        self._update_flow_sign()
        self._update_out_and_inflows_for_vertices()
        self._verify_mass_balance()
        #print('Mass balance verified updated')
        self._t=t
        self._tSample=tSample
        stdout.flush()

        G['dtFinal']=t
        #G['htts0']=htts0
        #G['keepRBCs0']=keepRBCs0
        #G['numbers0']=numbers0
        #G['vf0']=vf0
        #G['v0']=v0
        if self._analyzeBifEvents:
            G['rbcsMovedPerEdge']=self._rbcsMovedPerEdge
            G['edgesMovedRBCs']=self._edgesWithMovedRBCs
            G['rbcMovedAll']=self._rbcMoveAll
        #G['iterFinalPlot']=tPlot
        G['iterFinalSample']=tSample
        G['BackUpCounter']=BackUpCounter
        #G['spacing']=self._spacing
        filename1='sampledict_BackUp_'+str(BackUpCounter)+'.pkl'
        filename2='G_BackUp'+str(BackUpCounter)+'.pkl'
        if self._capDil != 0:
            analyzeDivBifs={}
            analyzeDivBifs['upstreamDivBifs']=self._upstreamDivBifs
            analyzeDivBifs['divBifEdges']=self._DivBifEdges
            analyzeDivBifs['analyzeDivBifEvents']=self._analyzeDivBifEvents
            analyzeDivBifs['upstreamDivBifsCount']=self._upstreamDivBifsCount
            analyzeDivBifs['upstreamDivBifsNotDiv']=self._upstreamDivBifsNotDiv
            vgm.write_pkl(analyzeDivBifs,'analyzeDivBifs_BackUp'+str(BackUpCounter)+'.pkl')
        #if doPlotting:
        #    filename= 'iter_'+str(int(round(tPlot+1)))+'.vtp'
        #    filenamelist.append(filename)
        #    timelist.append(tPlot)
        #    self._plot_rbc(filename)
        #    g_output.write_pvd_time_series('sequence.pvd', 
        #                                   filenamelist, timelist)
        if doSampling:
            self._sample()
            #Convert deaultUnits to 'pBC' ['mmHG']
            for v in G.vs:
                if v['pBC'] != None:
                    v['pBC']=v['pBC']/self._scaleToDef
                v['pressure']=v['pressure']/self._scaleToDef
            self._sample_average()
            g_output.write_pkl(self._sampledict, 'sampledict.pkl')
            g_output.write_pkl(self._sampledict,filename1)
            #g_output.write_pkl(self._transitTimeDict, 'TransitTimeDict.pkl')
            #g_output.write_pvd_time_series('sequenceSampling.pvd',
            #                                   filenamelistAvg, timelistAvg)
        #G['spacing']=self._spacing
        vgm.write_pkl(G, 'G_final.pkl')
        vgm.write_pkl(G,filename2)
        # Since Physiology has been rewritten using Cython, it cannot be
        # pickled. This class holds a Physiology object as a member and
        # consequently connot be pickled either.
        #g_output.write_pkl(self, 'LSHTD.pkl')
        #self._timelist = timelist[:]
        #self._filenamelist = filenamelist[:]
        #self._filenamelistAvg = filenamelistAvg[:]
        #self._timelistAvg = timelistAvg[:]

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
        scaleToDef=self._scaleToDef
        #Convert default units to ['mmHG']
        pressure=np.array([1/scaleToDef]*G.vcount())*G.vs['pressure']
        
        for eprop in ['flow', 'v', 'htt', 'htd','nRBC','nRBCint','effResistance']:
            if not eprop in sampledict.keys():
                sampledict[eprop] = []
            sampledict[eprop].append(G.es[eprop])
        for vprop in ['pressure']:
            if not vprop in sampledict.keys():
                sampledict[vprop] = []
            sampledict[vprop].append(pressure)
        if not 'time' in sampledict.keys():
            sampledict['time'] = []
        sampledict['time'].append(self._tSample)

    #--------------------------------------------------------------------------

    def _sample_average(self):
        """Averages the self._sampleDict data and writes it to disc.
        INPUT: sampleAvgFilename: Name of the sample average out-file.
        OUTPUT: None
        """
        sampledict = self._sampledict
        G = self._G
        if 'averagedCount' in sampledict.keys():
            avCount=sampledict['averagedCount']
        else:
            avCount = 0
        avCountNew=len(sampledict['time'])
        avCountE=np.array([avCount]*G.ecount())
        avCountNewE=np.array([avCountNew]*G.ecount())
        for eprop in ['flow', 'v', 'htt', 'htd','nRBC','nRBCint','effResistance']:
            if eprop+'_avg' in G.es.attribute_names():
                G.es[eprop + '_avg'] = (avCountE*G.es[eprop+'_avg']+ \
                    avCountNewE*np.average(sampledict[eprop], axis=0))/(avCountE+avCountNewE)
            else:
                G.es[eprop + '_avg'] = np.average(sampledict[eprop], axis=0)
            #if not [eprop + '_avg'] in sampledict.keys():
            #    sampledict[eprop + '_avg']=[]
            sampledict[eprop + '_avg']=G.es[eprop + '_avg']
        avCountV=np.array([avCount]*G.vcount())
        avCountNewV=np.array([avCountNew]*G.vcount())
        for vprop in ['pressure']:
            if vprop+'_avg' in G.vs.attribute_names():
                G.vs[vprop + '_avg'] = (avCountV*G.vs[vprop+'_avg']+ \
                    avCountNewV*np.average(sampledict[vprop], axis=0))/(avCountV+avCountNewV)
            else:
                G.vs[vprop + '_avg'] = np.average(sampledict[vprop], axis=0)
            #if not [vprop + '_avg'] in sampledict.keys():
            #    sampledict[vprop + '_avg']=[]
            sampledict[vprop + '_avg']=G.vs[vprop + '_avg']
        sampledict['averagedCount']=avCount + avCountNew
        G['averagedCount']=avCount + avCountNew
        #g_output.write_vtp(G, sampleAvgFilename, False)


    #--------------------------------------------------------------------------
    #@profile
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
            x = abs(AA.solve(self._b, tol=self._eps/10000000000000000000, accel='cg')) # abs required, as (small) negative pressures may arise
        elif method == 'iterative2':
         # Set linear solver
             ml = rootnode_solver(A, smooth=('energy', {'degree':2}), strength='evolution' )
             M = ml.aspreconditioner(cycle='V')
             # Solve pressure system
             #x,info = gmres(A, self._b, tol=self._eps, maxiter=50, M=M, x0=self._x)
             #x,info = gmres(A, self._b, tol=self._eps/10000000000000, maxiter=50, M=M)
             x,info = gmres(A, self._b, tol=self._eps/10000, maxiter=50, M=M)
             if info != 0:
                 print('SOLVEERROR in Solving the Matrix')
             test = A * x
             res = np.array(test)-np.array(self._b)
        self._x = x
        ##self._res=res

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
        for i in range(G.vcount()):
            if G.vs[i]['flowSum'] > 5e-4 and i not in G['av'] and i not in G['vv']:
                print('')
                print(i)
                print(G.vs['flowSum'][i])
                #print(self._res[i])
                print('FLOWERROR')
                for j in G.adjacent(i):
                    print(G.es['flow'][j])

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
    #--------------------------------------------------------------------------
    
    def _nRBC_mean_outflow(self,posRBC0,edge,deltaT,steps=20):
        """Computes the average contribution of one RBC leaving a vessel to the 
        tube hematocrit during one time step
        """
        length=edge['length']
        vRBC=edge['v']
        lRBC=edge['minDist']
        if edge['sign'] == 1:
            tlim = (length-0.5*lRBC-posRBC0)/vRBC
        else:
            tlim = (posRBC0-0.5*lRBC)/vRBC
        nRBCvals=[]
        if tlim < 0: #RBC already started leaving vessel in the last time step
            tlim = 0 
        for i in range(1,steps):
            t=i*deltaT/steps
            if t > tlim:
                nRBCvals.append(np.max([1-((t-tlim)*vRBC)/lRBC,0]))
            else:
                nRBCvals.append(1)

        nRBCInt=np.mean(nRBCvals)
       
        return nRBCInt
                
    #--------------------------------------------------------------------------
    
    def _nRBC_mean_inflow(self,posRBC1,edge,deltaT,steps=20):
        """Computes the average contribution of one RBC just entering a vessel to the 
        tube hematocrit during one time step
        """
        length=edge['length']
        vRBC=edge['v']
        lRBC=edge['minDist']
        if edge['sign'] == 1:
            tlim = (posRBC1+0.5*lRBC)/vRBC
        else:
            tlim = (length+0.5*lRBC-posRBC1)/vRBC
        if tlim < 0:
            print('WARNING tlim negative')
        nRBCvals=[]
        if tlim >= deltaT:
            tlim = 0
        for i in range(1,steps):
            t=i*deltaT/steps
            if t > tlim:
                nRBCvals.append(np.min([(t-tlim)*vRBC/lRBC,1]))
            else:
                nRBCvals.append(0)

        nRBCInt=np.mean(nRBCvals)
       
        return nRBCInt
                

                   
