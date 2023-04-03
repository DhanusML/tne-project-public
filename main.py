from gradProj import gradProj
from funcs import readData, readFlowData
from greedy import greedy


if __name__ == "__main__":
    network = 'SiouxFalls'
    metaDataDict, arcs, nodes = readData(f'./data/{network}/{network}_net.tntp')
    #metaDataDict, arcs, nodes = readData('./ChicagoSketch_net.tntp')
    numNodes = metaDataDict['numNodes']
    numLinks = metaDataDict['numLinks']
    print(metaDataDict)

    odMat = readFlowData(f'./data/{network}/{network}_trips.tntp')

    #  gradProj(nodes, arcs, odMat, numNodes, numLinks)
    greedy(nodes, arcs, odMat, numNodes, numLinks)

