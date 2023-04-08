from gradProj import gradProj
from funcs import readData, readFlowData, msa, frankWolfe
from greedy import greedy
import matplotlib.pyplot as plt
import time
import numpy as np


if __name__ == "__main__":
    #network = 'SiouxFalls'
    network = 'ChicagoSketch'
    metaDataDict, arcs, nodes = readData(f'./data/{network}/{network}_net.tntp')
    odMat = readFlowData(f'./data/{network}/{network}_trips.tntp')

    numNodes = metaDataDict['numNodes']
    numLinks = metaDataDict['numLinks']
    print(metaDataDict)

    '''
    msa_start = time.time()
    data_msa = msa(nodes, arcs, odMat, numNodes, numLinks, False)
    msa_end = time.time()
    print("msa: ", msa_end-msa_start)
    np.savetxt('./results/data_msa.txt', data_msa)

    metaDataDict, arcs, nodes = readData(f'./data/{network}/{network}_net.tntp')
    fw_start = time.time()
    data_fw = frankWolfe(nodes, arcs, odMat, numNodes, numLinks, False)
    fw_end = time.time()
    print("fw: ", fw_end-fw_start)
    np.savetxt('./results/data_fw.txt', data_fw)
    '''


    metaDataDict, arcs, nodes = readData(f'./data/{network}/{network}_net.tntp')
    gp_start = time.time()
    data_gp = gradProj(nodes, arcs, odMat, numNodes, numLinks, False)
    gp_end = time.time()
    print("gp: ", gp_end-gp_start)
    np.savetxt('./results/newnew/data_gp.txt', data_gp)

    #  reset the arcs #
    metaDataDict, arcs, nodes = readData(f'./data/{network}/{network}_net.tntp')
    greedy_start = time.time()
    data_greedy = greedy(nodes, arcs, odMat, numNodes, numLinks, False)
    greedy_end = time.time()
    print("greedy: ", greedy_end-greedy_start)
    np.savetxt('./results/newnew/data_greedy.txt', data_greedy)



    '''
    plt.plot(data_gp[:,0], data_gp[:,1], label='gradProj')
    plt.plot(data_greedy[:,0], data_greedy[:,1], label='greedy')
    plt.plot(data_msa[:,0], data_msa[:,1], label='msa')
    plt.plot(data_fw[:,0], data_fw[:,1], label='frankWolfe')
    plt.xlabel('iterations')
    plt.ylabel('relativity gap')
    plt.title(network)
    '''

    #plt.legend()
    #plt.show()

