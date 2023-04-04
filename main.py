from gradProj import gradProj
from funcs import readData, readFlowData, msa, frankWolfe
from greedy import greedy
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":
    network = 'SiouxFalls'
    metaDataDict, arcs, nodes = readData(f'./data/{network}/{network}_net.tntp')
    odMat = readFlowData(f'./data/{network}/{network}_trips.tntp')
    #metaDataDict, arcs, nodes = readData('./ChicagoSketch_net.tntp')

    numNodes = metaDataDict['numNodes']
    numLinks = metaDataDict['numLinks']
    print(metaDataDict)

    msa_start = time.time()
    data_msa = msa(nodes, arcs, odMat, numNodes, numLinks, True)
    msa_end = time.time()

    metaDataDict, arcs, nodes = readData(f'./data/{network}/{network}_net.tntp')
    fw_start = time.time()
    data_fw = frankWolfe(nodes, arcs, odMat, numNodes, numLinks, True)
    fw_end = time.time()


    metaDataDict, arcs, nodes = readData(f'./data/{network}/{network}_net.tntp')
    gp_start = time.time()
    data_gp = gradProj(nodes, arcs, odMat, numNodes, numLinks, True)
    gp_end = time.time()

    #  reset the arcs #
    metaDataDict, arcs, nodes = readData(f'./data/{network}/{network}_net.tntp')
    greedy_start = time.time()
    data_greedy = greedy(nodes, arcs, odMat, numNodes, numLinks, True)
    greedy_end = time.time()


    print("msa: ", msa_end-msa_start)
    print("fw: ", fw_end-fw_start)
    print("gp: ", gp_end-gp_start)
    print("greedy: ", greedy_end-greedy_start)

    plt.plot(data_gp[:,0], data_gp[:,1], label='gradProj')
    plt.plot(data_greedy[:,0], data_greedy[:,1], label='greedy')
    plt.plot(data_msa[:,0], data_msa[:,1], label='msa')
    plt.plot(data_fw[:,0], data_fw[:,1], label='frankWolfe')
    plt.xlabel('iterations')
    plt.ylabel('relativity gap')
    plt.title(network)

    plt.legend()
    plt.show()

