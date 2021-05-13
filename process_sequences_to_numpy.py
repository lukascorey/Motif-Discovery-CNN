import numpy as np

def code(nt): 
    if nt.lower() == "a":
        return 0
    elif nt.lower() == "c": 
        return 2
    elif nt.lower() == "t":
        return 1
    elif nt.lower() == "g": 
        return 3

with open("sequences_all_14k_for_lukas_3_30_2021.txt", "r") as f: 
    three_prime = f.read().splitlines()[1:]


wt3 = np.ndarray(shape=(len(three_prime), 31,4), dtype=int, order = 'F')
mut3 = np.ndarray(shape=(len(three_prime), 31,4), dtype=int, order = 'F')

for i in range(len(three_prime)): 
    line = three_prime[i].split("\t")
    for j in range(31):
        wt3[i, j, code(line[34][j+86])] = 1 # 23
        mut3[i, j, code(line[35][j+86])] = 1 # 24

wf =  open("wt3_30.npy", "wb+")
np.save(wf, wt3)
mf = open("mut3_30.npy", "wb+")
np.save(mf, mut3)
print(mut3.shape)
