from funcs import labelCorrecting

def updateFlows(oldPath, newPath, change, arcs):
    oldPath.flow -= change
    newPath.flow += change

    for arcNum in oldPath.path:
        arcs[arcNum-1].flow -= change

    for arcNum in newPath.path:
        arcs[arcNum-1].flow += change


def getConvergenceParams(nodes, arcs, odMat, numNodes, numLinks):
    tstt = getTSTT(arcs)
    sptt = getSPTT(nodes, arcs, odMat, numNodes, numLinks)
    gap = (tstt/sptt) - 1

    return gap, tstt, sptt


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
            if j==i or odMat[i][j]==0:
                continue
            destination = j+1

            print(origin, destination, end='-->')
            for pathObj in p_hat[(origin, destination)]:
                print(pathObj.path,
                      [round(arcs[x-1].flow, 2) for x in pathObj.path],
                      round(pathObj.flow,2),
                      round(sum([arcs[x-1].time for x in pathObj.path]),2),
                      sep='--', end='|')
                print()
