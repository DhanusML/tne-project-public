from dataStructures import Arc, PathFlowObj
from funcs import labelCorrecting



def gradProj(nodes, arcs, odMat, numNodes, numLinks):
    p_hat = initializeGradProj(nodes, arcs, odMat, numNodes, numLinks)
    printPathFlow(p_hat, arcs, odMat)
    numZones = odMat.shape[0]
    iteration = 0

    while iteration < 10000:
        for i in range(numZones):
            origin = i+1

            _, pathArcDict = labelCorrecting(
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
                        if len(a_hat)==0:
                            flowUpdate = pathObj.flow
                        else:
                            denom = sum([arcs[x-1].der for x in a_hat])
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

        tstt = getTSTT(arcs)
        sptt = getSPTT(nodes, arcs, odMat, numNodes, numLinks)

        gap = tstt/sptt - 1

        print("iteration: ", iteration)
        print("tstt: ", tstt)
        print("sptt: ", sptt)
        print("gap: ", gap)

        iteration += 1


def updateFlows(oldPath, newPath, change, arcs):
    oldPath.flow -= change
    newPath.flow += change

    for arcNum in oldPath.path:
        arcs[arcNum-1].flow -= change

    for arcNum in newPath.path:
        arcs[arcNum-1].flow += change



def getTSTT(arcs):
    tstt = 0
    for arc in arcs:
        tstt += (arc.flow*arc.time)

    return tstt

def getSPTT(nodes, arcs, odMat, numNodes, numLinks):
    sptt = 0
    numZones = odMat.shape[0]
    for i in range(numZones):
        origin = i+1
        _, pathArcDict = labelCorrecting(
            origin, nodes, arcs, numNodes, numLinks
        )
        for j in range(numZones):
            if j==i:
                continue
            destination = j+1
            demand = odMat[i][j]
            p_star = pathArcDict[destination]
            sptt += demand*getPathTime(p_star, arcs)

    return sptt



def getPathTime(path, arcs):
    tot_time = 0
    for arcNum in path:
        tot_time += arcs[arcNum-1].time

    return tot_time


def addPath(p_star, paths):
    pathList = [x.path for x in paths]
    if p_star not in pathList:
        paths.append(PathFlowObj(p_star, 0))

    else:  # put the shortest path to the top
        thisIndex = pathList.index(p_star)
        paths[thisIndex], paths[-1] = paths[-1], paths[thisIndex]



def printPathFlow(p_hat, arcs, odMat):
    numZones = odMat.shape[0]

    for i in range(numZones):
        origin = i+1

        for j in range(numZones):
            if j==i:
                continue
            destination = j+1

            print(origin, destination, end='-->')
            for pathObj in p_hat[(origin, destination)]:
                print(pathObj.path,
                      [arcs[x-1].flow for x in pathObj.path],
                      round(pathObj.flow,2),
                      round(sum([arcs[x-1].time for x in pathObj.path]),2),
                      sep='--', end='|')
                print()



def initializeGradProj(nodes, arcs, odMat, numNodes, numLinks):
    numZones = odMat.shape[0]
    p_hat = {}

    for i in range(numZones):

        origin = i+1

        _, pathArcDict = labelCorrecting(
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
