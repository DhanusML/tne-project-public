import numpy as np
from dataStructures import Arc, PathFlowObj
from funcs import labelCorrecting
from pathBasedHelpers import getConvergenceParams, getPathTime,\
    printPathFlow


def greedy(nodes, arcs, odMat, numNodes, numLinks, verbose=False):
    data = []
    p_hat = initializeGreedy(nodes, arcs, odMat, numNodes, numLinks)
    gap = float('inf')
    delta = 0#float('inf')
    numZones = odMat.shape[0]
    iteration = 0
    start_time = time.time()
    time_spent = 0

    while gap>1e-4 and time_spent<8*3600:
        for i in range(numZones):
            origin = i+1

            _, pathArcDict = labelCorrecting(origin, nodes, arcs,
                                             numNodes,numLinks)

            for j in range(numZones):
                if j==i or odMat[i][j]==0:
                    continue
                destination = j+1

                p_star = pathArcDict[destination]

                if p_star not in [x.path for x in p_hat[(origin, destination)]]:
                    p_hat[(origin, destination)].append(PathFlowObj(p_star,0))

                ##  perform path flow adjustments ##
                p_hat[(origin, destination)] = greedyLoop(
                    p_hat[(origin, destination)],
                    arcs, odMat[i][j]
                )


        innerIter = 0
        maxInnerIter = 1000
        flowChange = 0

        while innerIter < maxInnerIter:
            innerIter += 1
            flowChange = 0
            for key in p_hat.keys():
                i = key[0] - 1
                j = key[1] - 1
                if innerIter%100 == 0:
                    times = [getPathTime(x.path, arcs) for x in p_hat[key]]
                    delta = max(times) - min(times)

                if delta>gap/2:
                    flowChange += 1

                    ##  perform path flow adjustments acc. Alg1  ##
                    p_hat[key] = greedyLoop(p_hat[key], arcs, odMat[i][j])

                    ##  update link flows and travel times  ##
                    for pathObj in p_hat[key]:
                        for arcNum in pathObj.path:
                            arcs[arcNum-1].updateArcTime()
                            arcs[arcNum-1].updateArcTimeDer()

            if flowChange == 0:
                break

        ##  convergence check  ##
        gap, tstt, sptt = getConvergenceParams(nodes, arcs, odMat,
                                               numNodes, numLinks)

        if verbose:
            #printPathFlow(p_hat, arcs, odMat)
            print("iteration: ", iteration)
            print("tstt: ", tstt)
            print("sptt: ", sptt)
            print("gap: ", gap)
            print()
        data.append([iteration, gap])
        iteration += 1
        time_spent = time.time()-start_time

    data = np.array(data)
    return data


def greedyLoop(paths, arcs, demand):
    #  v = [getPathTime(x.path, arcs) for x in paths]
    #  s = [sum([arcs[t-1].der for t in p.path]) for p in paths]

    v = []
    s = []
    c = []

    for pathObj in paths:
        thisV = getPathTime(pathObj.path, arcs)
        v.append(thisV)
        thisS = sum([arcs[t-1].der for t in pathObj.path])
        s.append(thisS)
        c.append(thisV - pathObj.flow*thisS)

    sortedIndices = np.argsort(c)
    paths = [paths[i] for i in sortedIndices]
    s = [s[i]+1e-12 for i in sortedIndices]
    c = [c[i] for i in sortedIndices]

    currentIndex = 0
    currentS = s[currentIndex]
    currentC = c[currentIndex]

    B = 1/(currentS*demand)
    C = currentC/(currentS*demand)
    w_bar = (1 + C)/B

    newPaths = [paths[currentIndex]]
    currentIndex = 1

    while currentIndex<len(paths):
        currentS = s[currentIndex]
        currentC = c[currentIndex]

        if currentC >= w_bar:
            break

        C += currentC/(currentS*demand)
        B +=  1/(currentS*demand)
        w_bar = (1+C)/B

        newPaths.append(paths[currentIndex])
        currentIndex += 1


    for i, pathObj in enumerate(newPaths):
        newFlow = (w_bar - c[i])/s[i]
        if newFlow < 0:
            #raise(ValueError("newFlow should be non-negative"))
            pass
        pathObj.newFlow = newFlow

    for pathObj in paths:
        if pathObj not in newPaths:
            pathObj.newFlow = 0


    for pathObj in paths:
        if pathObj.newFlow != pathObj.flow:
            for arcNum in pathObj.path:
                arcs[arcNum-1].flow += (pathObj.newFlow-pathObj.flow)
                arcs[arcNum-1].updateArcTime()
                arcs[arcNum-1].updateArcTimeDer()


    for pathObj in paths:
        pathObj.flow = pathObj.newFlow

    return newPaths


def initializeGreedy(nodes, arcs, odMat, numNodes, numLinks):
    numZones = odMat.shape[0]
    p_hat = {}

    for i in range(numZones):
        origin = i+1

        _, pathArcDict = labelCorrecting(
            origin, nodes, arcs,
            numNodes, numLinks
        )

        for j in range(numZones):
            if i==j or odMat[i][j]==0:
                continue
            destination = j+1
            p_hat[(origin, destination)] = []
            p_star = pathArcDict[destination]
            p_hat[(origin, destination)].append(PathFlowObj(p_star, odMat[i][j]))

    for i in range(numZones):
        origin = i+1

        for j in range(numZones):
            if i==j or odMat[i][j]==0:
                continue
            destination = j+1

            for pathObj in p_hat[(origin, destination)]:
                for arcNum in pathObj.path:
                    arcs[arcNum-1].flow += pathObj.flow

    for arc in arcs:
        arc.updateArcTime()
        arc.updateArcTimeDer()

    return p_hat
