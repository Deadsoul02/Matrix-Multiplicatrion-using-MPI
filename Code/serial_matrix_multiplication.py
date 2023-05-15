import time

N = 100  # size of matrices

A = [[i+j for j in range(N)] for i in range(N)]
B = [[i-j for j in range(N)] for i in range(N)]
C = [[0 for j in range(N)] for i in range(N)]

# multiply matrices
start = time.time()
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]
end = time.time()

# print result
for i in range(N):
    for j in range(N):
        print(C[i][j], end=' ')
    print()

# calculate time difference
print(f"Time taken: {end - start} seconds")