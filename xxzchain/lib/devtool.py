import numpy as np
import matplotlib.pyplot as plt
from ..core import *
#__all__ = ['load_system','preset','to_operator']


def cleanup(scope):
    l = list(vars(scope).keys())
    for var in l:
        vars(scope)[var] = None
        del vars(scope)[var]

def plot_spin(spin, size):
    spin = np.array(list('{0:b}'.format(spin).zfill(size)))=='1'
    target = np.arange(size)[spin]
    empty = np.arange(size)[np.logical_not(spin)]
    a = plt.figure(figsize = [15 - 14*np.exp(1/5*(1-target.shape[0])),1])
    plt.axhline(y = 0,color = 'black')
    plt.plot(target,[0 for _ in target],'o-',c= 'black',markersize = 20)
    plt.plot(empty,[0 for _ in empty],'o-',c = 'black',mew = 2,mfc = 'white',markersize = 20)
    plt.xlim([-1,size])
    plt.xticks([])
    plt.yticks([])

class OperatorGroup:
    def __init__(self, system,  name=None):
        self._operators = {}
        self.system = system
        self.name = None
        self.description  = None
        self._index = {}
        self._i = 0
        if not name is None:
            if not name in system._Operator:
                self.name = name
                system._Operator[name] = self
            else:
                #name = 'Op{}'.format(len(system._op))
                print("same operator name exist. set_name() please.")
    def save(self, force =False):
        if self.name == None:  raise ValueError("This operator has no name. please run 'set_name()' first.")

        saver = self.system.saver.require_group('/operator')
        if not self.system.saver.is_exist('/operator/{}'.format(self.name)):
            gsaver = saver.create_group(self.name)
            saver[self.name].attrs['name'] = self.name.encode()
        else:
            gsaver = saver[self.name]
        for op in self._operators:
            self._operators[op].save(force = force, saver = gsaver)
            #gsaver[op].attrs['index'] =
        gsaver.attrs['des'] = str(self.description).encode()
        gsaver.attrs['group'] = True


    def load(system, name):
        if system.saver.is_exist('/operator/{}'.format(name)):
            temp = OperatorGroup(system, name)
            loader = system.saver['/operator/{}'.format(name)]
            for op in loader:
                self[op] = Operator.load(system, op, loader)
        return temp

    def __setitem__(self, key, value):
        '''make operator as member of group '''
        if key in self._operator:
            raise KeyError("key already exist! if you want to delete, use 'del ogj[key]'.")
        if isinstance(value, Operator):
            self._operators[key] = value
            self._index[self._i] = key
            self._i += 1
            value.set_group(self)
        else:
            raise ValueError("Only Operator can be grouped")
        return

    def __delitem__(self, key):
        for i in self._index:
            if self._index[i] == key:
                target = i
        for i in range(target+1, self._i):
            self._index[i] = self._index[i+1]
        del self._index[self._i], self._operater[key]
        self._i -=1


    def __getitem__(self, key):
        if key in self._operater:
            return self._operators[key]
        elif isinstance(key, int):
            return self._operater[self._index[key]]
        else:
            raise NotImplemented()


    def __call__(self, key):
        return self.__getitem__(key)
