class Arc:
    def __init__(self, arcNum, arcData):
        self.arcNum = arcNum
        self.headNode = arcData[0]
        self.tailNode = arcData[1]
        self.capacity = arcData[2]
        self.length = arcData[3]
        self.ffTime = arcData[4]
        self.b = arcData[5]
        self.power = arcData[6]
        self.speedLimit = arcData[7]
        self.toll = arcData[8]
        self.type = arcData[9]

        self.flow = 0
        self.aonFlow = 0
        self.time = self.ffTime
        self.der = 1e-13

    def updateArcTime(self):
        newTime = self.getTime(self.flow)
        self.time = newTime

    def updateArcTimeDer(self):
        self.der = self.getArcTimeDer()

    def getArcTimeDer(self):
        const = (self.ffTime*self.b*self.power)/self.capacity
        #factor = (1+self.b*(self.flow/self.capacity))**(self.power-1)

        if self.power==0 or self.flow==0:
            return 0

        else:
            factor = (self.flow/self.capacity)**(self.power -1)

        return const*factor

    def getTime(self, flow):
        if self.flow==0 or self.power==0:
            newTime = self.ffTime
            return newTime
        else:
            newTime = self.ffTime*(1+self.b*((flow/self.capacity)**self.power))
        return newTime



class Node:
    def __init__(self, nodeNum):
        self.node = nodeNum
        self.adjNodes = []
        self.adjArcs = []
        self.prevNode = -1
        self.prevArc = None
        self.label = float('inf')

    def resetLabel(self):
        self.prevNode = -1
        self.prevArc = -1
        self.label = float('inf')

    def addAdjNode(self, num):
        self.adjNodes.append(num)

    def addAdjArc(self, num):
        self.adjArcs.append(num)


class PathFlowObj:
    def __init__(self, path=[], flow=0):
        self.path = path
        self.flow = flow
        self.newFlow = 0

    def setFlow(self, flow):
        self.flow = flow

