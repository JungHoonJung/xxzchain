import h5py
import os
import numpy as np
from ..core import *


__all__ = ['preset','load_system','to_operator']

def to_operator(system):
    '''decorator for make operator given function. System will hold
    given function below this decorator.'''
    if not isinstance(system, System):
        raise SyntaxError("use @to_operator(system).")
    class setter:
        def __init__(self, system):
            self.system = system
        def __call__(self, func):
            ufunc = inspect.getsource(func).split('\n')[1:]
            ufunc_name = ufunc[0][ufunc[0].find('def')+3:ufunc[0].find('(')].split()[0]
            if ufunc_name in self.system._function:
                raise NameError("already same name exist!")
            code = ''
            for line in ufunc:
                code += line+'\n'
            vect = 'import numpy as np\n@np.vectorize\n'+code
            saver = self.system.saver.require_group('function')
            function=saver.create_group(ufunc_name)
            function.attrs['code'] = code.encode()
            function.attrs['fid'] = self.system._fid
            self.system._fid+=1
            compiled = compile(vect, 'userfunction','exec')
            exec(compiled, self.system._function)
            del self.system._function['np'],self.system._function['__builtins__']
            temp = Operator(self.system, ufunc_name, prefix= self.system._function[ufunc_name])
            return temp

    wrapper = setter(system)
    return wrapper

def preset(system, globalscope):
    '''define 4 operators on given system and given scope'''
    globalscope['n_i'] = Operator(system, prefix=Operator.n_i)
    globalscope['b_i'] = Operator(system, prefix=Operator.annihilation)
    globalscope['b_i_dag'] = Operator(system, prefix= Operator.creation)
    globalscope['hopping'] = Operator(system, prefix= Operator.hopping)

def load_system(path, print_tree=True):
    '''Please indicate system saved folder. return is system object.
    Recommended way is use of 'load_system(path)'  '''
    if not path[0] =='/':
        path = os.path.abspath(path)

    if os.path.isfile(path+'.hdf5'):
        path = path + '.hdf5'

    if os.path.isfile(path):
        file = h5py.File(path)
        temp = System(file.attrs['size'], name = file.attrs['name'].decode())
        file.close()
        temp.load(path, print_tree)
        return temp
    else:
        raise FileExistsError("'{}' doesn't exist".format(path))
