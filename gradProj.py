import numpy as np
import time
from dataStructures import Arc, PathFlowObj
from funcs import labelCorrecting, labelSetting
from pathBasedHelpers import getConvergenceParams, getPathTime,\
    printPathFlow


def gradProj(nodes, arcs, odMat, numNodes, numLinks, verbose=False):
    data = []
    p_hat = initializeGradProj(nodes, arcs, odMat, numNodes, numLinks)
    #printPathFlow(p_hat, arcs, odMat)
    numZones = odMat.shape[0]
    iteration = 0
    gap = float('inf')
    start_time = time.time()
    time_spent = 0

    while gap>1e-4 and time_spent<24*3600:
        for i in range(numZones):
            origin = i+1

            _, pathArcDict = labelSetting(
                origin, nodes,
                arcs, numNodes, numLinks
            )

            for j in range(numZones):
                if j==i:
                    continue
                destination = j+1

                p_star = pathArcDict[destination]
                tau_star = getPathTime(p_star, arcs)

                addPath(p_star, p_hat[(origin, destination)])

                if len(p_hat[(origin, destination)])==1:
                    p_hat[(origin, destination)][0].flow = odMat[i][j]

                else:
                    for pathObj in p_hat[(origin, destination)][:-1]:
                        tau = getPathTime(pathObj.path, arcs)
                        a_hat = set(pathObj.path).symmetric_difference(set(p_star))
                        #denom = sum([arcs[x-1].der for x in a_hat])
                        denom = 1
                        flowUpdate = min(pathObj.flow,
                                         (tau-tau_star)/denom)


                        if flowUpdate<0:
                            raise(ValueError("flow change must be positive"))

                        updateFlows(pathObj, p_hat[(origin, destination)][-1],
                                    flowUpdate, arcs)


            for arc in arcs:
                arc.updateArcTime()
                arc.updateArcTimeDer()

        for key in p_hat.keys():  # remove paths with zero flow
            p_hat[key] = [x for x in p_hat[key] if x.flow != 0]

        gap, tstt, sptt = getConvergenceParams(nodes, arcs, odMat,
                                               numNodes, numLinks)

        if verbose:
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


def updateFlows(oldPath, newPath, change, arcs):
    oldPath.flow -= change
    newPath.flow += change

    for arcNum in oldPath.path:
        arcs[arcNum-1].flow -= change

    for arcNum in newPath.path:
        arcs[arcNum-1].flow += change


def addPath(p_star, paths):
    pathList = [x.path for x in paths]
    if p_star not in pathList:
        paths.append(PathFlowObj(p_star, 0))

    else:  # put the shortest path to the top
        thisIndex = pathList.index(p_star)
        paths[thisIndex], paths[-1] = paths[-1], paths[thisIndex]


def initializeGradProj(nodes, arcs, odMat, numNodes, numLinks):
    numZones = odMat.shape[0]
    p_hat = {}

    for i in range(numZones):

        origin = i+1

        _, pathArcDict = labelSetting(
            origin, nodes, arcs,
            numNodes, numLinks
        )

        for j in range(numZones):
            if i==j:
                continue
            destination = j+1
            p_hat[(origin, destination)] = []
            p_star = pathArcDict[destination]
            p_hat[(origin, destination)].append(PathFlowObj(p_star, odMat[i][j]))


    for i in range(numZones):
        origin = i+1

        for j in range(numZones):
            if j==i:
                continue
            destination = j+1

            for pathObj in p_hat[(origin, destination)]:
                for arcNum in pathObj.path:
                    arcs[arcNum-1].flow += pathObj.flow

    for arc in arcs:
        arc.updateArcTime()
        arc.updateArcTimeDer()

    return p_hat
