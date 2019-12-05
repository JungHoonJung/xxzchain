import xxzchain as xxz
import numpy as np
import sys

path = sys.argv[1]
Q = sys.argv[2]
num = sys.argv[3]
np.arange(5)

S = xxz.load_system(path)

H_fin         = S.get_operator('H_fin')[num]
H_ini         = S.get_operator('H_ini')[num]
n   , N       = S.get_operator("n"), S.get_operator("n")
initial_basis = S.get_basis(Q,0)
print('initial basis : {} state'.format(len(initial_basis)))
S.Hamiltonian = H_fin
print("Hamiltonian : {}".format(S.Hamiltonian.name))

# get all energy eigenvalues of Nb = 7
energies = []
for k in S.range:
    for e in S.get_basis(Q,k).energy:
        energies.append(e)
assert len(energies) == len(S.get_basis(Q))
energies = np.array(energies)
np.save(path+"energy{}.npy".format(num), energies)
