import vgm
import numpy as np

from vgm import VascularGraph

def createTestGraph():
    """
    """
    G=vgm.VascularGraph(12)
    alpha=70*np.pi/180
    alpha2=np.arccos(2*np.cos(alpha))
    length=60
    diameter=6

    #Coordinates
    r=[
    #Inlet
    [0,0,0],#0
    [length,0,0],#1
    #Lower branch
    [np.cos(alpha2)*length+length,-1*np.sin(alpha2)*length,0],#2
    [np.cos(alpha2)*length+2*length,-1*np.sin(alpha2)*length,0],#3
    #upper branch
    [np.cos(alpha)*length+length,np.sin(alpha)*length,0],#4
    [2*np.cos(alpha)*length+length,2*np.sin(alpha)*length,0],#5
    [2*np.cos(alpha)*length+2*length,2*np.sin(alpha)*length,0],#6
    [2*np.cos(alpha)*length+length,0,0],#7
    [2*np.cos(alpha)*length+2*length,0,0],#8
    [3*np.cos(alpha)*length+2*length,np.sin(alpha)*length,0],#9
    #outlet
    [2*np.cos(alpha2)*length+2*length,0,0],#10
    [2*np.cos(alpha2)*length+3*length,0,0],#11
    ]

    G.vs['r']=r
    #Edges
    edgeTuples=[
    #Inlet
    (0,1), #0
    #Lower branch
    (1,2), #1
    (2,3), #2
    (3,10),#3
    #upper branch
    (1,4), #4
    (4,5), #5
    (5,6), #6
    (6,9), #7
    (4,7), #8
    (7,8), #9
    (8,9), #10
    (9,10),#11
    #outlet
    (10,11)#12
    ]

    G.add_edges(edgeTuples)
    G.es['length']=[length]*G.ecount()
    G.es['diameter']=[diameter]*G.ecount()
    G.vs['av']=[0]*G.vcount()
    G.vs['vv']=[0]*G.vcount()
    G.vs['pBC']=[None]*G.vcount()
    G.vs['rBC']=[None]*G.vcount()
    G.vs[0]['av']=1
    G.vs[11]['vv']=1
    G['av']=G.vs(av_eq=1).indices
    G['vv']=G.vs(vv_eq=1).indices
    G.vs[11]['pBC']=0
    G.vs[0]['rBC']=40

    return G
