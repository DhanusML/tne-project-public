import sys
import numpy as np
import matplotlib.pyplot as plt

# Set the default text font size
plt.rc('font', size=18)
# Set the axes title font size
plt.rc('axes', titlesize=18)
# Set the axes labels font size
plt.rc('axes', labelsize=18)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=16)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=16)
# Set the legend font size
plt.rc('legend', fontsize=16)
# Set the font size of the figure title
plt.rc('figure', titlesize=18)

plt.figure(figsize=(8,6))


network = sys.argv[1]
#network = "SiouxFalls"

algorithms = ['msa', 'fw', 'gp', 'greedy']


for alg in algorithms:
    data = np.loadtxt(f'./results/{network}/data_{alg}.txt')
    plt.plot(data[:,0], np.log10(data[:,1]), label=f'{alg}')



plt.xlabel('iterations')
plt.ylabel('$\log_{10}$(RG)')
plt.title(f'{network}')
plt.legend()
plt.savefig(f'{network}.png', dpi=300)
#plt.show()
