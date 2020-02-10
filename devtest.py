# %%
import xxzchain as xxz
# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
test = xxz.System(21)
# %%
test.initialize('test')
test.__doc__
# %%
a = test.get_operator('H_fin')
# %%
a.get_eigenvalue(test.get_basis(1))
# %%
H_ini = test.get_operator('H_ini')
H_ini.latex
test.get_basis(1,0).state
# %%
test.get_basis(1,0).state_set(1)
# %%
a.get_eigenvalue(test.get_basis(2))
# %%
te = xxz.State(test.get_basis(1,0),np.array([1]))
# %%
te
# %%
a(te)
# %%
b = a.latex
# %%
b
# %%
display(Math('\\sum_i^L{-t(b_i^\\dagger b_{i+1}+ b_{i+1}^\\dagger b_{i})}'))
# %%
xxz.preset(test, globals())
# %%
H_ini = xxz.Operator.Null(test)
t = 0.5
V = 2.0
t_ = 0.24
V_ = 0.24
for i in range(test.size):
    H_ini += -t*hopping.index(i, i+1) + V*(n_i.index(i)-1/2)@(n_i.index(i+1)-1/2)
    H_ini += -t_*hopping.index(i, i+2) + V_*(n_i.index(i)-1/2)@(n_i.index(i+2)-1/2)
# %%
H_ini.set_name('H_ini')
# %%
H_ini.latex = r'''H_{ini} = \sum_i^L -t\left(b^\dagger_{i+1}b_i + b^\dagger_{i}b_{i+1}  \right)
+V\left(n_i - \frac{1}{2}\right)\left(n_{i+1}  - \frac{1}{2}\right)\\
-t'\left(b^\dagger_{i+2}b_i + b^\dagger_{i}b_{i+2}  \right)
+V'\left(n_i - \frac{1}{2}\right)\left(n_{i+2}  - \frac{1}{2}\right)\\
(t=0.5,V=2.0, \quad t'=V'=0.24)'''
# %%
H_fin = xxz.Operator.Null(test)
t = 1.0
V = 1.0
t_ = 0.24
V_ = 0.24
for i in range(test.size):
    H_fin += -t *hopping.index(i, i+1) + V *(n_i.index(i)-1/2)@(n_i.index(i+1)-1/2)
    H_fin += -t_*hopping.index(i, i+2) + V_*(n_i.index(i)-1/2)@(n_i.index(i+2)-1/2)
# %%
H_fin.set_name('H_fin')
# %%
H_fin.latex = r'''H_{fin} = \sum_i^L -t\left(b^\dagger_{i+1}b_i + b^\dagger_{i}b_{i+1}  \right)
+V\left(n_i - \frac{1}{2}\right)\left(n_{i+1}  - \frac{1}{2}\right)\\
-t'\left(b^\dagger_{i+2}b_i + b^\dagger_{i}b_{i+2}  \right)
+V'\left(n_i - \frac{1}{2}\right)\left(n_{i+2}  - \frac{1}{2}\right)\\
(t=1.0,V=1.0, \quad t'=V'=0.24)'''
# %%
test.Hamiltonian = H_ini

# %%
H_ini.latex
# %%
zero_energy = H_ini.get_eigenvalue(test.get_basis(7,0))
# %%
energy = []
for k in range(test.size):
    for e in H_fin.get_eigenvalue(test.get_basis(7,k)):
        energy.append(e)

energy = np.array(energy)
# %%
def partition_f(H,beta):

    return np.exp(-H*beta)

def temper(func, beta):
    def wrapper(x):
        return func(x,beta)
    return wrapper
# %%
eff_T = []
E_set = []
for i in range(100):
    beta = 1/3*(i+1)
    partition = temper(partition_f,beta)
    Z = partition(energy).sum()
    eff_T.append(1/beta)
    E_set.append((energy*partition(energy)/Z).sum())
# %%
energy.sort()
# %%
zero_state = H_ini.get_eigenstates(test.get_basis(7,0))
# %%
plt.plot(energy)
# %%
plt.plot(zero_energy)
# %%
plt.plot(E_set)
# %%
E_set[0]
# %%
zero_energy = H_ini.get_eigenvalue(test.get_basis(7,0))
# %%
zero_energy[np.logical_and(zero_energy<-3.8,zero_energy>-3.9)]
# %%
zero_energy[244]
# %%
init_state = H_ini.get_eigenstates(test.get_basis(7,0))
# %%
Energy=H_fin.expectation(zero_state)
# %%
 Energy[np.logical_and(Energy<-3.8,Energy>-3.9)]
# %%
target= Energy[np.logical_and(Energy<-3.8,Energy>-3.9)][1]
# %%
for i,e in enumerate(Energy):
    if e== target:
        print(i,e)
        break
# %%
Energy_sort = Energy.copy()
# %%
Energy_sort.copy()
# %%
test.save()
# %%
plt.plot(Energy)
# %%

# %% markdown
# ###
# %%
t2 = zero_state[346]
# %%
t2.shape
# %%
t2 = xxz.State(test.get_basis(7,0),t2,'init4')
# %%
len(t2)
# %%
n = {}
for k in range(test.size):
    n[k] = xxz.Operator.Null(test)
    for i in range(test.size):
        for j in range(test.size):
            n[k] += 1/test.size*np.exp(-2j*np.pi/test.size*k*(i-j))*b_i_dag.index(i)@b_i.index(j)
# %%
N = {}
for k in range(test.size):
    N[k] = xxz.Operator.Null(test)
    for i in range(test.size):
        for j in range(test.size):
            N[k] += 1/test.size*np.exp(-2j*np.pi/test.size*k*(i-j))*n_i.index(i)@n_i.index(j)
# %%
n_k = []
for k in n:
    #n[k].get_matrix(test.get_basis(7,0))
    n_k.append(n[k].expectation(zero_state))
# %%
K = []
for k in n:
    K.append(k/test.size*2*np.pi  if k/test.size < 1/2 else -2*np.pi*(1-k/test.size))
# %%
plt.figure(figsize = [10,5])
plt.plot(K,np.array(n_k).real,'x')
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],[r'-\pi',r'-\pi/2',0,r'\pi/2',r'-\pi'])
plt.savefig("nk.png")
# %%
np.save('n_k1.npy',np.array(n_k))
# %%
N_k=[]
for k in N:
    N_k.append(N[k].expectation(zero_state))
# %%
plt.figure(figsize = [9,3])
plt.plot(K[1:],N_k[1:],'bX')
plt.plot([0],[0],'bX')
plt.ylim(0,0.4)
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],[r'-\pi',r'-\pi/2',0,r'\pi/2',r'-\pi'])
plt.savefig('largeNk.png')
# %%
init_state
# %%
my_equation = '\\frac{x^2}{2}'
# %%
from IPython.display import Latex
display(Latex(my_equation))
from IPython.display import Math
display(Math(my_equation))
# %%
zero_energy = H.get_eigenvalue(test.get_basis(7,0))
# %%
energy
# %%
E_set[4]
# %%
plt.plot(1/np.array(eff_T), E_set)
# %%
beta = 0.1
# %%

# %%
test.save()
# %%
len(test.get_basis(7,1))
# %%
E_set = []
for sector
# %%
import matplotlib.pyplot as plt
# %%
plt.plot(energy[0])
# %%
plt.plot(energy[3])
# %%
t
# %%
H_V = [xxz.Operator.Null(test) for i in range(10)]
for v in range(10):
    for i in range()
    H_V[v]
# %%
for q in range(test):
