from mpi4py import MPI
import numpy as np

N = 100# size of matrice
MASTER = 0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

A = np.empty((N, N), dtype=int)
B = np.empty((N, N), dtype=int)
C = np.empty((N, N), dtype=int)

# initialize matrices
if rank == MASTER:
    for i in range(N):
        for j in range(N):
            A[i][j] = i + j
            B[i][j] = i - j

# scatter matrix A
local_A = np.empty((N//size, N), dtype=int)
comm.Scatter(A, local_A, root=MASTER)

# broadcast matrix B
comm.Bcast(B, root=MASTER)

# multiply matrices
start = N//size*rank
end = N//size*(rank+1)
start_time = MPI.Wtime() # start timer
for i in range(start, end):
    for j in range(N):
        C[i][j] = 0
        for k in range(N):
            C[i][j] += local_A[i-start][k] * B[k][j]
end_time = MPI.Wtime() # end timer

# gather matrix C
comm.Gather(C[start:end], C, root=MASTER)

# print result and execution time
if rank == MASTER:
    # print("Result matrix:")
    # for i in range(N):
    #     for j in range(N):
    #         print(C[i][j], end=" ")
    #     print()
    req_time = end_time - start_time
    print(f"Execution time for {N}x{N} matrice: {req_time} seconds")

MPI.Finalize()
