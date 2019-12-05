#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xxzchain as xxz

import numpy as np


# In[2]:


system =  xxz.System(21)


# In[3]:


system.initialize('test')


# In[4]:


xxz.preset(system, globals())


# ### momentum distribution function

# <font size='5'>
# $$n(k) \equiv \frac{1}{L} \sum_{i,j}  e^{-k(i-j)}\hat b^\dagger_i \hat b_j$$

# In[5]:


n = xxz.OperatorGroup(system)
n.set_name('n')


# In[6]:


for k in system.range:
    temp = xxz.Operator.Null(system)
    for i in system.range:
        for j in system.range:
            temp += 1/system.size*np.exp(-2j*np.pi*k*(i-j))*b_i_dag.index(i)@b_i.index(j)
    n[k] = temp


# <font size='5'>
# $$N(k) \equiv \frac{1}{L} \sum_{i,j}  e^{-k(i-j)}\hat n_i \hat n_j$$

# In[7]:


N = xxz.OperatorGroup(system)
N.set_name('N')


# In[8]:


for k in system.range:
    temp = xxz.Operator.Null(system)
    for i in system.range:
        for j in system.range:
            temp += 1/system.size*np.exp(-2j*np.pi*k*(i-j))*n_i.index(i)@n_i.index(j)
    N[k] = temp


# In[9]:


system.save()


# <font size='5'>
# $$H_{nn} = hopping + repulsive. potential$$

# <font size='5'>
# $$H_{nn} = \sum_i -t(\hat b^\dagger_i \hat b_{i+1} + \textrm{H.c}) + V\left(\hat n_i -\frac{1}{2}\right)\left( \hat n_{i+1}- \frac{1}{2}\right)$$

# <font size='5'>
# $$H_{nnn} = \sum_i -t'(\hat b^\dagger_i \hat b_{i+2} + \textrm{H.c}) + V'\left(\hat n_i -\frac{1}{2}\right)\left( \hat n_{i+2}- \frac{1}{2}\right)$$

#

# <font size='5'>
# $$H_{ini} = H_{nn}(t=0.5, V = 2) + H_{nnn}(t'=V'=c)$$

# <font size='5'>
# $$H_{fin} = H_{nn}(t=1, V = 1) + H_{nnn}(t'=V'=c)$$

# In[10]:


nnn = xxz.Operator.Null(system)
for i in system.range:
    nnn+= -1*hopping.index(i,i+2) + 1*(n_i.index(i)-1/2)@(n_i.index(i+2)-1/2)
nnn.set_default(False)
nnn.set_name('nnn')


# In[11]:


ini = xxz.Operator.Null(system)
for i in system.range:
    ini+= -0.5*hopping.index(i,i+1) + 2*(n_i.index(i)-1/2)@(n_i.index(i+1)-1/2)
ini.set_default(False)
ini.set_name('ini')


# In[12]:


fin = xxz.Operator.Null(system)
for i in system.range:
    fin += -1*hopping.index(i,i+1) + 1*(n_i.index(i)-1/2)@(n_i.index(i+1)-1/2)
fin.set_default(False)
fin.set_name('fin')


# In[13]:


coef = []
for i in range(5):
    j = 2**i
    coef.append(0.02*j)
    coef.append(0.03*j)


# In[14]:


coef.insert(0,0)


# In[15]:


coef


# In[16]:


(nnn+ini).coef.shape


# In[17]:


H_ini = xxz.OperatorGroup(system)
H_fin = xxz.OperatorGroup(system)
H_ini.set_name('H_ini')
H_fin.set_name('H_fin')
for c in coef:
    if c:
        H_ini[c] = ini+c*nnn
        H_fin[c] = fin+c*nnn
    else:
        H_ini[c] = ini
        H_fin[c] = fin


# In[18]:


system.save()


# In[19]:



# In[ ]:
