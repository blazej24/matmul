import sys
import random

N = int(sys.argv[1])
NNZ_per_row = int(sys.argv[2])

print(str(N), str(N), str(N*NNZ_per_row), str(NNZ_per_row))

for _ in range(N * NNZ_per_row):
    print(round(random.uniform(1.2, 5931.3), 5), end=' ')
print()

idx = 0
for i in range(N):
    print(idx, end=' ')
    idx += NNZ_per_row
print(idx)
for i in range(N):
    for c in random.sample(range(N), NNZ_per_row):
        print(c, end=' ')
print()
