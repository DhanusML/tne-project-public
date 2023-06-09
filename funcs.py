import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from dataStructures import Arc, Node, PathFlowObj


def readData(path):
    metaDataDict = {'numNodes': -1,
                    'numLinks': -1,
                    'numZones': -1,
                    'firstThruNode': -1
                   }
    arcs = []

    with open(path) as f:
        line = f.readline().strip()
        while line != '<END OF METADATA>':
            if line.startswith('<NUMBER OF NODES>'):
                metaDataDict['numNodes'] = int(line.split('>')[-1])

            if line.startswith('<NUMBER OF LINKS>'):
                metaDataDict['numLinks'] = int(line.split('>')[-1])

            if line.startswith('<NUMBER OF ZONES>'):
                metaDataDict['numZones'] = int(line.split('>')[-1])

            if line.startswith('<FIRST THRU NODE>'):
                metaDataDict['firstThruNode'] = int(line.split('>')[-1])

            line = f.readline().strip()

        nodes = [Node(i+1) for i in range(metaDataDict['numNodes'])]
        line = f.readline().strip()

        while len(line) == 0:
            line = f.readline().strip()

        arcNum = 1
        for line in f:
            line = line.lstrip().rstrip(';\n').rstrip().split('\t')
            headNode, tailNode = int(line[0]), int(line[1])
            line[0], line[1], line[-1] =\
                headNode, tailNode, int(line[-1])

            for i in range(2, len(line)-1):
                line[i] = float(line[i])

            nodes[headNode-1].addAdjArc(arcNum)
            nodes[headNode-1].addAdjNode(tailNode)
            arcs.append(Arc(arcNum, line))
            arcNum += 1

    return metaDataDict, arcs, nodes


def readFlowData(path):
    flow = 0.0
    numZones = 0

    with open(path) as f:
        line = f.readline().strip()
        while line != "<END OF METADATA>":
            if line .startswith("<NUMBER OF ZONES>"):
                numZones = int(line.split(">")[-1])

            if line.startswith("<TOTAL OD FLOW>"):
                flow = float(line.split(">")[-1])

            line = f.readline().strip()

        odMat = np.zeros((numZones, numZones))

        line = f.readline().strip()
        for _ in range(numZones):
            while len(line)==0 or line[0]=='~':
                line = f.readline().strip()

            if line.startswith('Origin'):
                thisZoneNum = int(line.split()[-1])

            line = f.readline().strip().strip(';')

            while len(line) != 0 and not line.startswith('Origin'):

                tempLine = line.strip().split(';')
                for keyVals in tempLine:
                    keyVals = keyVals.split(':')
                    keyVals = [int(keyVals[0]), float(keyVals[1])]
                    toZone = keyVals[0]
                    demand = keyVals[1]
                    odMat[thisZoneNum-1][toZone-1] = demand
                line = f.readline().strip().strip(';')

    return odMat


def getAdjMat(arcs, numNodes):
    mat = np.zeros((numNodes, numNodes))
    for arc in arcs:
        headNodePos = arc.headNode - 1
        tailNodePos = arc.tailNode - 1
        ffTime = arc.ffTime

        mat[headNodePos][tailNodePos] = ffTime
    return mat


def getIncidenceMat(arcs, numNodes, numLinks):
    mat = np.zeros((numLinks, numNodes))
    for arc in arcs:
        headNodePos = arc.headNode - 1
        tailNodePos = arc.tailNode -1
        arcPos = arc.arcNum -1

        mat[arcPos][headNodePos] = -1
        mat[arcPos][tailNodePos] = 1

    return mat.T


def getAdjListOfNodes(nodes):
    adjList = {}
    for i, node in enumerate(nodes):
        adjList[i+1] = node.adjNodes

    return adjList


def nxCheck(arcs, numNodes, numLinks):
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(1, numNodes+1)])

    for arc in arcs:
        headNode = arc.headNode
        tailNode = arc.tailNode
        weight = arc.ffTime
        G.add_edge(headNode, tailNode, weight=weight)

    print(G)
    A = nx.adjacency_matrix(G)
    B = nx.incidence_matrix(G, oriented=True)
    adjList = {}
    for i in range(1, numNodes+1):
        adjList[i] = list(G.neighbors(i))

    return A.todense(), B.todense(), adjList


def nxGetGraph(arcs, numNodes, numLinks):
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(1, numNodes+1)])

    for arc in arcs:
        headNode = arc.headNode
        tailNode = arc.tailNode
        weight = arc.ffTime
        G.add_edge(headNode, tailNode, weight=weight)

    return G


def getPaths(origin, nodes):
    pathDict = {}
    pathArcDict = {}

    for destination in nodes:
        pathDict[destination.node] = []
        pathArcDict[destination.node] = []

        thisNode = destination

        while(thisNode.node != origin):
            pathDict[destination.node].insert(0, thisNode.node)
            pathArcDict[destination.node].insert(0, thisNode.prevArc)
            thisNode = nodes[thisNode.prevNode-1]

        pathDict[destination.node].insert(0, origin)

    return pathDict, pathArcDict


def labelCorrecting(origin, nodes, arcs, numNodes, numLinks):
    #  initialization
    for i in range(numNodes):
        nodes[i].resetLabel()

    nodes[origin-1].label = 0
    nodes[origin-1].prevNode = origin
    sel = [origin]

    #  main loop
    while len(sel):
        i = sel.pop()
        for arc in nodes[i-1].adjArcs:
            thisArc = arcs[arc-1]
            j = thisArc.tailNode
            if nodes[j-1].label > nodes[i-1].label + thisArc.time:
                nodes[j-1].label = nodes[i-1].label + thisArc.time
                nodes[j-1].prevNode = i
                nodes[j-1].prevArc = arc
                if j not in sel:
                    sel.append(j)

    #for node in nodes:
    #    print(node.node, node.label)

    pathDict, pathArcDict = getPaths(origin, nodes)
    # print(pathDict)
    return pathDict, pathArcDict


def labelSetting(origin, nodes, arcs, numNodes, numLinks, firstThruNode=1):
    #  Dijkstra's

    #  initialization
    for i in range(numNodes):
        nodes[i].resetLabel()

    unVisited = set(range(1, numNodes+1))
    nodes[origin-1].label = 0
    nodes[origin-1].prevNode = origin

    #  main loop
    while(len(unVisited)):
        i = min(unVisited, key= lambda x: nodes[x-1].label)
        unVisited.remove(i)
        if i<firstThruNode:
            continue
        for arc in nodes[i-1].adjArcs:
            thisArc = arcs[arc-1]
            j = thisArc.tailNode
            if nodes[j-1].label > nodes[i-1].label + thisArc.time:
                nodes[j-1].label = nodes[i-1].label + thisArc.time
                nodes[j-1].prevNode = i
                nodes[j-1].prevArc = arc

    pathDict, pathArcDict = getPaths(origin, nodes)
    return pathDict  , pathArcDict
    # return nodes

def checkPaths(path1, path2, numNodes):
    for i in range(1, numNodes+1):
        if path1[i] != path2[i]:
            return False

    return True


def initMSA(nodes, arcs, odMat, numNodes, numLinks):
    numZones = odMat.shape[0]
    for i in range(numZones):
        origin = i+1
        pathDict, pathArcDict = labelSetting(
            origin, nodes, arcs,
            numNodes, numLinks
        )

        for j in range(numZones):
            destination = j+1
            for arcNum in pathArcDict[destination]:
                thisArc = arcs[arcNum-1]
                thisArcHead = thisArc.headNode
                thisArcTail = thisArc.tailNode
                thisArcDemand = odMat[i][j]
                thisArc.aonFlow += thisArcDemand

    tot1 = 0
    for arc in arcs:
        tot1+=arc.aonFlow
    #print("init", tot1)

    return arcs



def msa(nodes, arcs, odMat, numNodes, numLinks, verbose=False):
    arcs = initMSA(nodes, arcs, odMat, numNodes, numLinks)
    k = 1
    gap = float('inf')
    numZones = odMat.shape[0]

    data = []
    start_time = time.time()
    time_spent = 0
    while gap > 1e-4 and time_spent<8*3600:
        for arc in arcs:
            arc.flow = arc.aonFlow/k + (1-1/k)*arc.flow
            arc.updateArcTime()
            arc.aonFlow = 0

        for i in range(numZones):
            origin = i+1
            pathDict, pathArcDict = labelSetting(
                origin, nodes, arcs,
                numNodes, numLinks
            )

            for j in range(numZones):
                destination = j+1
                for arcNum in pathArcDict[destination]:
                    arcs[arcNum-1].aonFlow += odMat[i][j]
                    '''
                    thisArc = arcs[arcNum-1]
                    thisArcHead = thisArc.headNode
                    thisArcTail = thisArc.tailNode
                    thisArcDemand = odMat[i][j]
                    thisArc.aonFlow += thisArcDemand
                    '''

        gap = getRelGap(arcs)
        if verbose:
            print("iteration: ", k)
            #gap = getAEC(arcs, odMat)
            print("gap: ", gap)
            print()
        data.append([k, gap])
        k += 1
        time_spent = time.time()-start_time

    return np.array(data)



def frankWolfe(nodes, arcs, odMat, numNodes, numLinks, verbose=False):
    arcs = initMSA(nodes, arcs, odMat, numNodes, numLinks)
    k = 1
    gap = float('inf')
    data = []
    numZones = odMat.shape[0]

    start_time = time.time()
    time_spent = 0
    while gap > 1e-4 and time_spent<8*3600:
        if k==1:
            eta = 1

        else:
            eta = bisection(arcs)

        for arc in arcs:
            arc.flow = arc.aonFlow*eta + (1-eta)*arc.flow
            arc.updateArcTime()
            arc.aonFlow = 0

        for i in range(numZones):
            origin = i+1
            pathDict, pathArcDict = labelSetting(
                origin, nodes, arcs,
                numNodes, numLinks
            )

            for j in range(numZones):
                destination = j+1
                for arcNum in pathArcDict[destination]:
                    arcs[arcNum-1].aonFlow += odMat[i][j]
                    '''
                    thisArc = arcs[arcNum-1]
                    #thisArcHead = thisArc.headNode
                    #thisArcTail = thisArc.tailNode
                    thisArcDemand = odMat[i][j]
                    thisArc.aonFlow += thisArcDemand
                    '''
        gap = getRelGap(arcs)
        if verbose:
            print("iteration: ", k)
            #gap = getAEC(arcs, odMat)
            print("gap: ", gap)
            print()
        data.append([k, gap])
        k += 1
        time_spent = time.time()-start_time

    return np.array(data)

def getRelGap(arcs):
    tstt = 0
    sptt = 0
    for arc in arcs:
        tstt += arc.flow*arc.time
        sptt += arc.aonFlow*arc.time

    gap = tstt/sptt - 1

    return gap


def getAEC(arcs, odMat):
    tstt = 0
    sptt = 0
    for arc in arcs:
        tstt+=arc.flow*arc.time
        sptt += arc.aonFlow*arc.time

    aec = (tstt - sptt)/sum(odMat.reshape(-1))
    return aec


def bisection(arcs):
    epsilon = 1e-3
    eta_l = 0
    eta_u = 1

    while (eta_u-eta_l) > epsilon:
        eta = 0.5*(eta_l+eta_u)
        diff = 0

        for arc in arcs:
            conv = arc.aonFlow*eta + arc.flow*(1-eta)
            conv_time = arc.getTime(conv)
            diff += conv_time*(arc.aonFlow-arc.flow)

        if diff > 0:
            eta_u = eta

        else:
            eta_l = eta

    #print("eta: ", eta)
    #print("diff: ", diff)
    return eta


def getPathTime(path, arcs):
    time = 0
    for arcNum in path:
        time += arcs[arcNum-1].time

    return time


if __name__ == "__main__":
    network = 'SiouxFalls'
    metaDataDict, arcs, nodes = readData(f'./data/{network}/{network}_net.tntp')
    #metaDataDict, arcs, nodes = readData('./ChicagoSketch_net.tntp')
    numNodes = metaDataDict['numNodes']
    numLinks = metaDataDict['numLinks']
    print(metaDataDict)

    odMat = readFlowData(f'./data/{network}/{network}_trips.tntp')
    #odMat = readFlowData('./ChicagoSketch_trips.tntp')

    #pathDict, pathArcDict = labelCorrecting(1, nodes, arcs, numNodes, numLinks)

    #print(pathDict)
    #print(pathArcDict)
    #data1 = msa(nodes, arcs, odMat, numNodes, numLinks)
    #data2 = frankWolfe(nodes, arcs, odMat, numNodes, numLinks)
    gradProj(nodes, arcs, odMat, numNodes, numLinks)

    #plt.plot(data1[10:, 0], data1[10:, 1], label="msa")
    #plt.plot(data2[10:, 0], data2[10:, 1], label="fw")
    #plt.legend()
    #plt.show()
    #plt.savefig("combined_chicago.png")
