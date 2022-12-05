import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.linalg import svd

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
random.seed(rank)
#print("World Size: " + str(world_size) + "   " + "Rank: " + str(rank))

def cond(x, matrixSize):
    return (x >= 0) and (x < matrixSize)

def matrixConstructor(p, matrixSize):
    """ At any step the memory matrix is updated"""
    mtx = np.zeros((matrixSize,matrixSize), dtype=float)
    mtx[p] = 1
    counter = 1
    q = p
    pool = neighboring(q,mtx,matrixSize)
    while len(pool) > 0:
        for q in pool: mtx[q] = -1
        q = random.choice(pool)
        #q = pool[rng.integers(len(pool))]
        mtx[q] = 1
        pool = neighboring(q,mtx,matrixSize)
        counter += 1
    return (mtx, counter/np.sqrt(matrixSize))

def neighboring(entry, mtx, matrixSize):
    """  It defines de neighboring points $N(p_t)$ of an entry $p_t$. 
    The memory matrix mtx must vanishes at any entry corresponding to a 
    neighboring point """

    i, j = entry[0], entry[1]
    pool = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
    return [q for q in pool if cond(q[0],matrixSize) and cond(q[1],matrixSize) and mtx[q]==0]

def samplingEV(samplingSize, matrixSize):
    
    evs = []
    for i in range(samplingSize):
        A = matrixConstructor((random.choice(range(matrixSize)),random.choice(range(matrixSize))), matrixSize)[0]
        evs = np.concatenate([evs,np.linalg.svd(np.matmul(A,np.transpose(A)), full_matrices=False, compute_uv=False)])        
        
    eigenvalues = []
    for i in range(matrixSize):
        ev = evs[i:matrixSize*samplingSize:matrixSize]
        eigenvalues.append(np.mean(ev))
        
    return np.array(eigenvalues)

def samplingEVS(samplingSize, matrixSize):
    
    evs = []
    for i in range(samplingSize):
        A = matrixConstructor((random.choice(range(matrixSize)),random.choice(range(matrixSize))), matrixSize)[0]
        evs = np.concatenate([evs,np.linalg.svd(np.matmul(A,np.transpose(A)), full_matrices=False, compute_uv=False)])        
        
    return evs

def samplingEVrank(evs, samplingSize, matrixSize):
    
    eigenvalues = []
    for i in range(matrixSize):
        ev = evs[i:matrixSize*samplingSize:matrixSize]
        eigenvalues.append(np.mean(ev))
        
    return np.array(eigenvalues)

def computeLoads(N):
    workloads = [N//size for i in range(size)]
    for i in range(N%size):
        workloads[i] += 1
    rini = 0
    for i in range(rank):
        rini += workloads[i]
    rend = rini + workloads[rank]

    return (rini,rend)


# User definitions
matrixSize   = 1000
samplingSize = 2000

rini, rend = computeLoads(samplingSize)

t = time.time()
evs = samplingEVS(rend-rini, matrixSize)
print(f"{time.time() - t:e}")
#print(f"{len(evs)}, rank = {rank}")

if rank == 0:
    EVS = np.zeros(matrixSize*samplingSize,dtype=float)
    EVS[rini*matrixSize:rend*matrixSize] = evs
    #print("\n", EVS)
    
    for i in range(1,size):
        sdata = np.zeros(2,dtype=int)
        comm.Recv(sdata, source=i, tag=0)
        #print(f"I rank = {rank} recieve ini = {sdata[0]} end = {sdata[1]} from rank = {i}.")
        
        evs = np.zeros(matrixSize*(sdata[1]-sdata[0]),dtype=float)
        comm.Recv(evs, source=i, tag=1)
        #print(evs, "from rank =", i)
        EVS[sdata[0]*matrixSize:sdata[1]*matrixSize] = evs
        #print("\n", EVS)

else:
    #print(f"{rini} - {rend}, rank = {rank}")
    sdata = np.array([rini,rend],dtype=int)
    #print(sdata)
    comm.Send(sdata, dest=0, tag=0)
    comm.Send(evs, dest=0, tag=1)
    #print(evs, "Im rank =", rank)

    
    
if rank == 0:
    
    eigenvalues = samplingEVrank(EVS, samplingSize, matrixSize)
    #print(f"{time.time() - t:e}")
    
    xx = range(len(eigenvalues))
    plt.scatter(xx, eigenvalues, s = 5.1)
    plt.xlim(-1, 20)
    plt.grid(True)
    plt.show()
    plt.savefig("plot.png")
    
    
    
#t = time.time()
#eigenvalues = samplingEV(samplingSize, matrixSize)
#print(f"{time.time() - t:e}")