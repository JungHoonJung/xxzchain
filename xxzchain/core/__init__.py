#!/usr/bin/env python
# coding: utf-8

# In[1]:
__all__ = ['System','Basis','Operator', 'State','Subsystem','OperatorGroup']#,'load_system']

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import inspect
from copy import deepcopy
#from IPython.display import Markdown, display
import h5py
import warnings
import logging
#from ..lib.devtool import OperatorGroup
#import cupy as cp


## In[2]:
LogFormatter = logging.Formatter("%(asctime)s '%(name)-10s' : [%(levelname)8s] %(message)s (ln : %(lineno)4s)")

class System:
    """A environment object of whole calculation.

    Parameters
    ----------
    size : ``integer``
        Size of the new system.
    name : ``string``
        Description of parameter `name`.
    dtype : ``np.dtype``
        Description of parameter `dtype`.


    Please make sure your system's size that you want to use. Support 1 ~ 32 lattice size.
    You can see the summery of your system through 'System.tree'.

    .. note::
        In many calculation, we represent spin configuration as binary.
        Generally, 4bytes-integer(``int``) is consist of 32 bits. It's the reason why this package support system size below 32.

    Attributes
    ----------
    range : ``range`` (`built-in`)
        Equivalent to ``range(size)``.
    Odtype : ``np.complex128``
        Data type of operator's elements.

    """
    #'''A environment object of quantum calculation.
#
#    Please make sure your system's size and symmetry that you want to use.
#    Support 1 ~ 32 lattice size.
#    You can see the summery of your system through 'System.tree'
#    '''


    def __init__(self, size, name = None, dtype = np.float64,**kwarg):
        self.size = size
        self.max = 1<<size
        self.range = range(self.size)
        self.name = None
        if name is None:
            self.name = "lattice{}".format(size)
        #self.sample = np.zeros([size],dtype =np.bool)

        #Data types
        self.dtype = dtype
        self.Odtype = np.complex128

        #logging
        self.logger = logging.getLogger("System")
        self.logger.setLevel(logging.WARNING)
        self._logmode = 'warning'
        for key in kwarg:
            if key == 'log':
                fileHandler = logging.FileHandler(kwarg[key])
                fileHandler.setFormatter(LogFormatter)
                self.logger.setLevel(logging.DEBUG)
                self.logger.addHandler(fileHandler)
                self._logmode = 'debug'
                #logging.basicConfig(filename = kwarg[key], format='%(asctime)s %(name)-10s : [%(levelname)-8s]|ln : %(lineno)s > %(message)s', level = logging.DEBUG)
                self.logger.info('System started. (size = {}, name = {}, dtype = {})'.format(self.size, self.name, self.dtype))
            else:
                KeyError("'{}'".format(key))


        #storage
        self.__basis = None
        self._Operator = {}
        self._State = {}
        self._function = {}
        self._Subsystem = {}
        self._fid = 0

        #status for system
        self.__initialized = False
        self.__path = None
        self.__H = None
        self.__symmetry = 'Q' # (spin conserving, traslational sym, parity, spin inversion)

        # TODO: for class 'Subsystem'
        self.parent = None

        #save/load & Initializer
        self.saver = Saver(self)
        self.initializer = Initializer(self)




    ############      tree        ################
    def tree():
        #doc =
        def fget(self):
            if not self.__initialized:
                print("System must be initialized. please make sure your path and file with 'initialize(path)'")
                return
            print("Hardcore boson system")
            path = '-----' if self.path is None else os.path.abspath(self.path)
            print("Name         : {}".format(self.name))
            print("Saved path   : {}".format(path))
            print("System size  : {}\n".format(self.size))
            print("Referenced Basis :   (None means full)")
            print('\t|- Sector {} '.format(self.__basis.symmetry))
            print('\t-----\n')

            H = '-----' if self.__H is None else "'{}'".format(self.__H)

            print("Defined Operators :")
            i = 0
            for op in self._Operator:
                print('\t|- {}'.format(op))
                if i>5:
                    print('\t|- ...')
                    break
                i+=1
            print('\t-----')
            print("\tHamiltonian  : {}\n".format(H))
            if self._State:
                print("Defined States :")
                i = 0
                for st in self._State:
                    print('\t|- {}'.format(st))
                    if i>5:
                        print('\t|- ...')
                        break
                        i+=1
                print('\t-----\n')
            if self._function:
                print("Defined Functions :")
                i = 0
                for st in self._function:
                    print('\t|- {}'.format(st))
                    if i>5:
                        print('\t|- ...')
                        break
                        i+=1
                print('\t-----\n')
            if self.saver.is_exist('data'):
                i = 0
                print("Stored Data :")
                for st in self.saver['data']:
                    print('\t|- {}'.format(st))
                    if i>5:
                        print('\t|- ...')
                        break
                        i+=1
                print('\t-----\n')
            #return self._tree
        #def fset(self, value):
        #    self._tree = value
        #def fdel(self):
        #    del self._tree
        return locals()

    tree = property(**tree())
    """Short summary of current system.


    .. note::
        If you run :py:meth:`xxzchain.core.System.initialize`, you will see output of this property.
    """

    @property
    def path(self): return self.__path
    @path.getter
    def path(self): return self.__path

    #for safety, didn't implement path.setter.

    def set_path(self, path):
        if self.__path is None:
            self.__path = path
        else:
            warnings.warn("System.path cannot be modified. 'set_path' is ignored.", RuntimeWarning)

    #####################   symmetry  ################## unused

    @property
    def symmetry(self):
        return self.__symmetry

    @symmetry.getter
    def symmetry(self):
        print("Current symmetry : ")
        if self.__symmetry[0]:
            print("\t|- Spin conserving : Q")
        if self.__symmetry[1]:
            print("\t|- Translational symmetry : K")
        if self.__symmetry[3]:
            print("\t|- Parity symmetry : P")
        if self.__symmetry[2]:
            print("\t|- Spin inversion symmetry : F")
        if not(self.__symmetry[0] or self.__symmetry[1] or self.__symmetry[2] or self.__symmetry[3]):
            print("\t|- No symmetry applied")
        print("\t---")

    ###############################################################################################

    ############# Hamiltonian     ##############
    def Hamiltonian():
        #doc = "The Hamiltonian property."
        def fget(self):
            if self.__H is None:
                return None
            if self.__H in self._Operator:
                return self._Operator[self.__H]
            elif len(self.__H.split("."))>1:
                return self._Operator[self.__H.split(".")[0]][".".join(self.__H.split(".")[1:])]

        def fset(self, name):
            if name is None:
                self.__H = None
                return
            for op in self._Operator:
                if op == name:
                    self.__H = name
                    self.logger.debug("Hamiltonian set as '{}'".format(self.__H))
                    return
                elif self._Operator[op] == name:
                    self.__H = op
                    self.logger.debug("Hamiltonian set as '{}'".format(self.__H))
                    return
            if type(name) == Operator:
                if name.name is None:
                    name.set_name('Hamiltonian')
                    self.__H = 'Hamiltonian'
                    self.logger.debug("Hamiltonian set as '{}'".format(self.__H))
                    return
                else:
                    self.logger.debug("Hamiltonian set as '{}'".format(self.__H))
                    return
            print("Cannot find Operator ({})".format(name))
            return
        def fdel(self):
            del self._Hamiltonian
        return locals()
    Hamiltonian = property(**Hamiltonian())
    """
    Hamiltonian operator of this system. Only operator which is defined on this system can be Hamiltonian.

    :type: :py:class:`xxzchian.core.Operator`



    .. note::
        :py:attr:`xxzchain.core.Basis.energy` and :py:meth:`xxzchain.core.State.time_evolving` or
        :py:meth:`xxzchain.core.State.time_evolving_states` will be affected by this property."""


    ############# initializer  ###############           will be modified
    def initialize(self, path = None, force = False):
        """Specific initializing method of this object.

        Parameters
        ----------
        path : ``str``
            The path that current system will be saved on.
        force : ``bool``
            If ``force`` == ``True``, system will ignore ``FileExistsError``.


        Our system use I/O system with HDF5 File Format to save calculation enviroment.
        every progress will be saved on ``path`` (i.e. Basis, Operator, State).

        .. note::
            If ``force`` argument is ``False`` and there is hdf5 file made by system earlier, this method will ask whether load that file or not.
            If you answer 'yes', this method will be equivalent to :py:func:`xxzchain.lib.load_system`.


        """
        #'''this method calculate of whole sector by given method.
        #every progress will be saved on path you set.
        #-->  calculate basis sector on spin number conserved basis sector.
        #other sector will be calculated when user call that basis sector.'''
        if path is None:
            if self.path is None:
                print("no preset path detected. system path set as {}".format(self.name))
                self.__path = self.name
                path = self.path
            else:
                path = self.path
        else:
            if self.path is None:
                self.__path  = path
        if not self.__path[-1] =='/':
            self.__path = path+'/'
            self.logger.debug('Current system will be saved at "{}"'.format(path))


        if self.__initialized:
            if not force:
                self.logger.warning("System already initialized!")
                return
            else:
                y = input("Previous progress will be lost. Continue?[Y/n] : ")
                if len(y) == 0 or not y.upper()[0] =='Y':
                    print('Initializing canceled.')
                    return
        else:
            if self.saver.isfile() and not force:
                y = ''
                while len(y)==0:
                    y = input("Previous save data exist. Will you load that data?[Y/n] : ")
                if y.upper()[0]=='Y':
                    if not path[-1] == '/': path+='/'
                    return self.load(path+self.name+'.hdf5')
                else:
                    print("Please make other path for save. Or use keyword 'force = True'.")
                    return

        os.makedirs(path,exist_ok=True)

        if (not os.path.isfile(path+self.name+'.hdf5')) or force:  ##########save initializing only for first
            self.saver._create()
            self.saver.attrs['name'] = self.name.encode()
            self.saver.attrs['size'] = self.size

            self.saver.attrs['H'] = str(self.__H).encode()
            self.saver.file.create_group('basis')
            self.logger.info("'{}' file created.".format(self.name+'.hdf5'))

        self.save()
        self.__initialized = True
        ## construct full sector
        basis  = self.get_full_sector()
        self.logger.debug("full basis tested.")
        ## find basis sector by given symmetry
        #self.initializer.Q(basis)
        del basis

        self.tree
        return


    ############## system ################
    def valid(self):
        return self.__initialized

    def validation(func):
        def caution(*arg, **kwargs):
            if not arg[0].valid():
                return print("Current system is empty or currupted. Please initialize or load first.")
            return func(*arg, **kwargs)
        return caution

    '''
    def set_name(self, name):
        self.name = name
    '''

    ####################### save ###################
    def save(self, name = None, data_array = None):
        """Save system or specific data into file.

        Parameters
        ----------
        name : ``str``
            Name of data which will be saved. After save data, you can access data with this name.
        data_array : array-like object
            Data which will be saved.


        """

            #self.checkfiles = path+'checkfiles'
            ########check files

        if name is not None:
            if data_array is None:
                print("Data is None")
                return
            saver = self.saver.require_group('data')
            saver.create_dataset(name, data = data_array)

        for op in self._Operator:
            self._Operator[op].save()
        self.saver.attrs['H'] = str(self.__H).encode()
        # -*- coding: utf-8 -*-
        for st in self._State:
            self._State[st].save()

        for sub in self._Subsystem:
            self._Subsystem[sub].save()

        self.saver.flush()

    def plot(self, data_path, *arg,**kwarg):
        """Plotting method for saved data. (*experimental*)

        Parameters
        ----------
        data_path : ``str``
            Name of data which you want to plot.
        ``*`` arg, ``**`` kwarg :
            Same ``arg`` and ``kwarg`` with matplotlib.pyplot.plot_.


        .. seealso::
            This method is using matplotlib.pyplot.plot_. If you need more detail for plotting, See reference.


        .. _matplotlib.pyplot.plot: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

        """
        if self.saver.is_exist('data/'+data_path):
            plt.plot(self.saver['data/'+data_path],*arg,**kwarg)
        if self.saver.is_exist(data_path):
            plt.plot(self.saver[data_path],*arg,**kwarg)

    def delete(self, name, force =False):
        """Delete saved data.

        Parameters
        ----------
        name : ``str``
            `name` of data which you want to delete.
        force : `boolean`
            If ``Ture``, method will delete data without asking.

        Raises
        -------
        ``KeyError``
            If the data which is given name doesn't exist.
        """
        '''this method delete your system that has been saved.
        PLEASE TAKE CARE FOR USING THIS METHOD.'''
        if not self.saver.is_exist('data/'+name):
            raise KeyError(str(name))
        if not force:
            y = input("'{}' will be removed on disk. Continue?[Y/n] : ".format(name))
            if len(y)==0 or not y.upper()=='Y':
                print("Delete cancelled")
                return
        del self.saver.file['data/{}'.foramt(name)]
        self.logger.info("file['data/{}'] deleted.".format(name))
        return

    def __del__(self):
        self.logger.debug("System deleted.")
        del self.logger
        target = list(vars(self).keys())
        for var in target:
            del vars(self)[var]
        return

    def load(self, path, print_tree= True):
        """Load system object from ``.hdf5`` (experimental).

        Parameters
        ----------
        path : `string`
            The File to read. Unfortunately, file-like object is not supported.
        print_tree : `bool, optional`
            Skipping show tree when the system loaded. If you try to load multiple system,
            tree will make bunch of text.

        Raises
        -------
        IOError
            If the input file path does not valid, or cannot be read.


        This method is not convenient to load system. We recommend using :py:func:`xxzchain.lib.load_system`
        """


        ########check files
        self.saver.open(path)
        slash = 0
        while path.find('/',slash+1) >0:
            slash = path.find('/',slash+1)
        self.__path = path[:slash+1]
        self.__initialized=True
        ##########load system
        self.size = self.saver.attrs['size']
        self.name = self.saver.attrs['name'].decode()
        self.__basis = self.get_full_sector()
        fn_ = self.saver.get('function', None)
        op_ = self.saver.get('operator',None)
        st_ = self.saver.get('state',None)
        if not fn_ is None:
            for fn in fn_:
                code = fn_[fn].attrs['code'].decode()
                vect = 'import numpy as np\n@np.vectorize\n'+code
                compiled = compile(vect, 'userfunction','exec')
                exec(compiled, self._function)
                del self._function['np'],self._function['__builtins__']
                if self._fid<fn_[fn].attrs['fid']:
                    self._fid = fn_[fn].attrs['fid']

        if not op_ is None:
            for op in op_:
                self._Operator[op] = Operator.load(self, op)
        if not st_ is None:
            for st in st_:
                self._State[st] = State.load(self, st)

        H = self.saver.attrs['H'].decode()
        if not H == 'None' and self._Operator.get(H, False):
            self.Hamiltonian = H

        print('Current system loaded by "{}"'.format(os.path.abspath(path)))
        if print_tree:
            self.tree



    ################# basis ################
    #@validation
    def get_full_sector(self):
        """Access to full sector of this system..

        Returns
        -------
        :py:class:`xxzchain.core.Basis`
            Basis object of full Hilbert space.

        """

        basis = Basis(self)
        state = np.arange(self.max)
        address = {i:i for i in state}
        counts = self.max
        basis.set_basis(state, address,1<<self.size)
        self.__basis = basis
        basis.data = self.saver['/basis']
        return basis

    #@validation
    def _basis(self):
        '''return last referenced basis'''
        if self.__basis:
            return self.__basis
        else:
            raise ValueError("There is no basis referenced.")

    #@validation
    def get_basis(self, *arg, **kwarg):
        """Get basis of given simultaneous eigenvalue sector.

        Parameters
        ----------
        *arg : ``Integer``
            Simultaneous eigenvalues of basis which you want to get in order (N[, K[, P[, X]]]).
        **kwarg : 'key( ``Operator`` ) = value( ``eigenvalue`` )'
            Specific kind of eigenvalues.

            i.e.)
                ``get_basis(N = 1)``, ``get_basis(n = 1, k = 0)``, ``get_basis(k = 0, p = 1)``


        Returns
        -------
        :py:class:`xxzchain.core.Basis`
            Basis object to manage specific Hilbert space sector.


        .. note::
            This method automatically search basis within data.
            if given sector is not found, then system will calculate.

            positional argument : eigenvalues in order of (N, K, P, X)

            keywork argument    : N = n, K = k, P = p, X = x (small and capital letter can be used for identifying each eigenvale.)

        """




        #validation
        if not self.valid():  return print("Current system is empty or currupted. Please initialize or load first.")

        if len(arg)>4 or len(kwarg)>4: raise KeyError("argument number is allowed only under 4.")
        null = (-1,-1,0,0)
        symbol = 'NKPX'
        ### Follow symmetry order QKFP and make symmetry tuple
        if len(kwarg)==0 and len(arg) == 0:
            return self.get_full_sector()
        if len(arg) == 1 and arg[0] == -1:
            return self.get_current_basis()
        if arg and kwarg or len(arg)>4 or len(kwarg)>4:
            raise KeyError("Too many values are given")
        if arg == null:
            return self.get_full_sector()
        if arg:
            symmetry = arg + null[len(arg):]
        elif kwarg:
            symmetry = list(null)
            for i in kwarg:
                ch = symbol.find(i.upper())
                if ch>=0:
                    symmetry[ch]  = kwarg[i]
                else:
                    raise AttributeError("Symmetry has no property : {}".format(i))

        self.logger.debug("given argument : {}".format(symmetry))
        #symmetry condtion check
        if symmetry[0]>self.size or symmetry[0]<-1:
            raise IndexError("given N = {} is out of bounds for current system size {}.".format(symmetry[0],self.size))
        if symmetry[1]>self.size or symmetry[1]<-1:
            raise IndexError("given K = {} is out of bounds for current system size {}.".format(symmetry[1],self.size))
        if symmetry[2]>1 or symmetry[2]<-1:
            raise IndexError("given P = {} is out of bounds for eigenvalue of spin inversion.".format(symmetry[2]))
        if symmetry[3]>1 or symmetry[3]<-1:
            raise IndexError("given X = {} is out of bounds for eigenvalue of parity.".format(symmetry[3]))

        self.logger.debug("called basis : {}".format(symmetry))
        basis = Basis(self, *symmetry)
        if not self.saver.is_exist('basis/({},{},{},{})'.format(*symmetry)):
            self.initializer.calculate(symmetry)
        basis.load(*symmetry)
        self.logger.debug("basis loaded. : {}".format(symmetry))
        self.__basis = basis
        return basis

    ## unused
    #@validation
    #def print_sector(self):
    #    for sector in self.__basis.keys():
    #        print(sector)
    #
    #@validation
    #def get_current_basis(self):
    #    return self.__basis


    def print_spin(self, value):
        """Print spin-configuration to binary representation.

        Parameters
        ----------
        value : ``Integer``
            Representation of spin-configuration.



        This method will print spinstate with 0(null site, down-spin),1(particle site, up-spin) of give integer format state onto console.
        If you put vanishing state as argument, method will print 'Vanishing'
        """
        try:
            print("{0:b}".format(value).zfill(self.size))
        except:
            print("Vanishing")

    def get_operator(self, op_name):
        """Getting operator with its name.

        Parameters
        ----------
        op_name : ``str``
            Operator's name that you've set. You can find list of operator with :py:meth:`xxzchain.core.System.operator_list`.

        Returns
        -------
        :py:class:`xxzchain.core.Operator`
            The operator which is named as ``st_name``.

        Raises
        -------
        KeyError
            If there is no operator that has name as ``op_name``.


        .. warning::
            To access to operator with operators name. before that, the operator name must be set on system.
        """

        return self._Operator[op_name]

    def get_data(self, key):
        """Getting data with its key.

        Parameters
        ----------
        key : ``str``
            Data's name when you save it.

        Returns
        -------
        ``np.ndarray``
            Data which is named as ``key``.

        Raises
        -------
        KeyError
            If there is no data that has name as ``key``.

        """
        if self.saver.is_exist('data/'+str(key)):
            return self.saver.get('data/'+str(key))[:]
        else:
            raise KeyError(key)

    def get_function(self, key):
        """Getting python function with its key.

        Parameters
        ----------
        key : ``str`` or ``Integer``
            Function's name when you save it.
            Or, order of saving.

        Returns
        -------
        function
            Python function object will be returned.

        Raises
        -------
        KeyError
            If there is no function that has name as ``key``. Or, out of index.

        """
        if isinstance(key, int):
            loader = self.saver['function']
            for function in loader:
                if loader[function].attrs['fid'] == key:
                    return self._function[function]
            raise KeyError(key)
        return self._function[key]

    def get_function_script(self, key):
        """Getting python function script with its key.

        Parameters
        ----------
        key : ``str`` or ``Integer``
            Function's name when you save it.
            Or, order of saving.

        Returns
        -------
        ``str``
            Function's source script.


        Raises
        -------
        KeyError
            If there is no function that has name as ``key``. Or, out of index.

        """
        loader = self.saver['function']
        if isinstance(key, int):

            for function in loader:
                if loader[function].attrs['fid'] == key:
                    print(loader[function].attrs['code'].decode())
                    return
            raise KeyError(key)
        for function in loader:
            if key == function:
                print(loader[function].attrs['code'].decode())
                return
        raise KeyError(key)


    ##################### no ues ##########################
    def load_operators_to_var(self, op_name_list=None, obj = None):
        """Stack given operators to namespace(variables).

        Parameters
        ----------
        op_name_list : list of ``str``
            The name list of operator which you want to load.
        obj : ``dict``
            The namespace or dictionary-like object to define operators. Default value is `globals()`


        """

        if op_name_list is None:
            op_name_list = list(self._Operator.keys())
        if obj is None:
            glob = globals()
        else:
            glob = vars(obj)
        for op in op_name_list:
            if glob.get(op, None) is None:
                glob[op] = self._Operator[op]
            else:
                print("'{}' is already defined on global variables".format(op))

    def get_state(self, st_name):
        """Getting operator with its name.

        Parameters
        ----------
        st_name : ``str``
            Name of state that you've set. You can find list of state with :py:meth:`xxzchain.core.System.state_list`.

        Returns
        -------
        :py:class:`xxzchain.core.State`
            The state which is named as ``st_name``.

        Raises
        -------
        KeyError
            If there is no operator that has name as ``st_name``.


        .. warning::
            To access to state with its name, the state must have name on system.
        """
        return self._State[st_name]

    def data_list(self):
        """List of name of data which is stored.

        Returns
        -------
        list
            List of name of data.

        """
        self.saver.ls('data',show_all=True)

    def operator_list(self):
        """List of name of operator which is stored.

        Returns
        -------
        list
            List of name of operator.

        """
        return list(self._Operator.keys())

    def state_list(self):
        """List of name of state which is stored.

        Returns
        -------
        list
            List of name of state.

        """
        return list(self._State.keys())

    def function_list(self):
        """List of name of function stored.

        Returns
        -------
        list
            List of name of function.

        """
        return list(self._function.keys())


#Subsystem

class Subsystem(System):
    def __init__(self, system, spin_range, name):
        super().__init__(len(spin_range), name = name)
        self.range = spin_range
        self.size = len(self.range)
        self.parent = system

    def initialize(self):
        self.spin_site = np.array([1<<i for i in self.range])
        self.saver.set_file(self.parent.saver.create_group(self.name))
        self.saver.attrs['size'] = self.size
        self.saver.attrs['range'] = self.range
        self.save()

    def load(self):
        raise NotImplementedError('not yet.')


# In[3]:


class Basis:
    '''Class 'basis' for Hilbert space sector of system.'''
    def __init__(self, system, N=None, K=None, P = None, X = None, symmetry = None):
        '''make instance of basis that hold given symmetry'''

        #dependency
        self.system = system
        self.size = system.size


        #property
        if not (symmetry is None):
            if len(symmetry) == 4:
                self.symmetry = np.array(symmetry)
                Np, Kp,Pp,Xp = self.symmetry
            else:
                raise KeyError("Symmetry is ambiguous. symmetry must have 4 component")
        else:
            Np = -1 if N is None else N
            Kp = -1 if K is None else K
            Pp = 0 if P is None else P
            Xp = 0 if X is None else X
            self.symmetry = np.array([Np,Kp,Pp,Xp], dtype = np.int8)

        self.N, self.K, self.P, self.X = self.symmetry
        self.__eigen = False
        self.temp = np.zeros([self.size],dtype = np.int8)
        self.__state, self.__address, self.__period, self.__counts = None,None,None,None
        self.data = None
        self.Pd_bar, self.Fd_bar = None, None

        self.path = "basis/({},{},{},{})".format(Np,Kp,Pp,Xp)
        self.find = np.vectorize(self._find)
        self.distance = np.vectorize(self.distance)
        self.period = np.vectorize(self._period)
        self.convert = np.vectorize(self.convert)
        self.__state_set = None

        self.__load = False

    def __len__(self):
        return len(self.state)

    def __repr__(self):
        if (self.symmetry==[-1,-1,0,0]).all():
            rep =  "<Full basis of lattice size {}".format(self.size)
        else:
            rep = "<Basis sector of lattice size {} with symmetry factor(N,K,P,X) = ({},{},{},{})".format(self.system.size, *self.symmetry)

        rep+='>'
        return rep


    ############ System  #################

    def set_basis(self, state,address, counts, period = None):
        self.__state = state
        self.__address = address
        self.__counts = counts
        self.__period = period
        if self.__period is None:
            self.__period = np.ones([len(self.__state)])
        self.__states = np.array(state)


    def address():
        #doc = "The address property."
        def fget(self):
            return self.__address
        def fdel(self):
            del self.__address
        return locals()
    address = property(**address())
    """return address dict of this basis"""

    @property
    def state(self):
        return self.__state

    @state.getter
    def state(self):
        return self.__state
    '''@state.setter
    def state(self, value):
        return'''

    def energy():
        doc = "The energy property."
        def fget(self):
            return self.system.Hamiltonian.get_eigenvalue(self)
        return locals()
    energy = property(**energy())

    def trace(self, operator, eigen=True):
        '''return Tr{ O } on this sector(s). basically, calculate eigenvalues and
        sum over. '''

        return operator.get_matrix(self)[np.diag_indices(len(self.state))].sum()


    def as_state(self):
        return State(self, np.ones([len(self.state)],dtype = self.system.Odtype))


    def _find(self, state):
        return self.__address.get(state,-1)

    def state_set(self, x):
        if self.__state_set is None:
            state_set = self.data['state_set'][:]
            self.__state_set={item[0]:item[np.logical_not(item == -1)][1:] for item in state_set}
        return self.__state_set.get(x,-1)



    @np.vectorize
    def _tran(x, size):
        if x == -1:
            return -1
        return (x>>(size-1))*((1)-(1<<size))+2*x

    def distance(self, target):
        if self.symmetry[1] == -1:
            raise EnvironmentError("'distance' method only supported for translational eigenbasis")
        return self.__distance[target]

    def _period(self, target):
        if target == -1:
            return -1
        return self.__period[target]

    def convert(self, x):
        address = self.find(x)
        if address == -1:
            return -1,0

        coef = 1
        if self.K*2 == self.system.size:
            coef *= -1
        elif self.K>0:
            coef = coef*np.exp(np.pi*2*1j*self.K*self.distance(x)/self.system.size)

        if self.P == -1:
            if self.Pd_bar is None:
                self.Pd_bar = {i:True for i in self.data['Pd_bar'][:]}
            if x in self.Pd_bar:
                coef *=-1

        if self.X == -1:
            if self.Xd_bar is None:
                self.Xd_bar = {i:True for i in self.data['Xd_bar'][:]}
            if x in self.Xd_bar:
                coef *= -1
        return address, coef


    def print_states(self, full_state = False, state_set = False):
        if full_state or len(self.state)<20:
            for i in self.state:
                s = "{:10d} : ".format(i)
                string = ''
                if state_set:
                    for state in self.state_set(i):
                        spin = "{0:b}".format(state)
                        spin = '|'+'0'*(self.system.size - len(spin))+spin+'>, '
                        string+= spin
                    string = string[:-2]
                else:
                    string +="{0:b}".format(i).zfill(self.system.size)
                print(s,string)
        else:
            for i in self.state[:3]:
                s = "{:10d} :".format(i)
                string = ''
                if state_set:
                    for state in self.state_set(i):
                        spin = "{0:b}".format(state)
                        spin = '|'+'0'*(self.system.size - len(spin))+spin+'>, '
                        string+= spin
                    string = string[:-2]
                else:
                    string +="{0:b}".format(i).zfill(self.system.size)
                print(s,string)
            print("...")
            for i in self.state[-3:]:
                if state_set:
                    for state in self.state_set(i):
                        spin = "{0:b}".format(state)
                        spin = '|'+'0'*(self.system.size - len(spin))+spin+'>, '
                        string+= spin
                    string = string[:-2]
                else:
                    string +="{0:b}  ".format(i).zfill(self.system.size)
                print(s,string)

    def load(self, N,K,P,X):
        '''load from hdf5 based on given symmetry factor.'''
        self.data = self.system.saver.get('basis/({},{},{},{})'.format(N,K,P,X))
        self.__state = self.data['state'][:]
        self.__address = { val : i for i, val in enumerate(self.__state)}
        self.__period = self.data['period'][:]
        if not len(self.__period) == len(self.__state):
            self.__period = np.ones([len(self.__state)])
        if not K == -1:
            state_set = self.data['state_set'][:]
            self.__state_set={item[0]:item[np.logical_not(item == -1)][1:] for item in state_set}

            for rs in self.__state_set:
                for s in self.__state_set[rs][1:]:
                    self.__address[s] = self.__address[rs]

            self.__period = []
            for s,p in self.data['period'][:]:
                if s in self.__address:
                    self.__period.append(p)

            self.__distance = { key:val for key, val in self.data['distance'][:]}
        self.__counts = self.data.attrs['counts']
        self.__load = True

    def get_path(self):
        return system.path+self.path


# In[22]:


class Operator:



    def __init__(self, system, name = None, prefix = None, group = None, use_gpu = False):
        '''Operator defined of specific system. One can define own operator using linear
        algebra. use @ symbol for multiply between operator. and also use numpy functions
        when setting coefficient with function.'''
        #print(identity(x))
        self.system = system
        self.group = group
        self.size = system.size
        self.name = None
        if not name is None:
            if not name in system._Operator:
                self.name = name
                system._Operator[name] = self
            else:
                #name = 'Op{}'.format(len(system._op))
                print("same operator name exist. set_name() please.")
        self.__default = True
        self._latex = ''
        if not prefix is None:
            self.acting = prefix
            self.__default = False
            if prefix == Operator.identity or prefix == Operator._table['I']:
                self._label = 'I'
            elif prefix == Operator.annihilation or prefix == Operator._table['A']:
                self._label = 'A'
            elif prefix == Operator.creation or prefix == Operator._table['C']:
                self._label = 'C'
            elif prefix == Operator.hopping or prefix == Operator._table['H']:
                self._label = 'H'
            elif prefix == Operator.n_i or prefix == Operator._table['N']:
                self._label = 'N'
            elif prefix == Operator.x_i or prefix == Operator._table['X']:
                self._label = 'X'
            else:
                for name in self.system._function:
                    if prefix == self.system._function[name]:
                        self._label = 'U'
                        fid = self.system.saver['function/'+name].attrs['fid']
                        if len(str(fid))==1:
                            self._label+='0{}'.format(fid)
                        else:
                            self._label+=str(fid)

        else:
            self.acting = Operator.identity
            self._label = 'I'
        #print(self._label)
        if self._label == '':
            raise KeyError
        self.struct = [self.acting]

        self.__coef = np.ones([1],dtype = self.system.Odtype)
        self.__initial = []
        self.matrix = None
        self._gpu = use_gpu
        self.__local = False

        #self.__diagonalized = False

    def _clear(self):
        del self.group, self.system
        for data in self.__initial:
            del self.system.saver[data]
        return

    def latex():
        doc = "The latex property."
        def fget(self):
            #self.system.display(Markdown('<font size ="4"> $$' + self._latex+'$$'))
            return self._latex
        def fset(self, value):
            self._latex = value
        def fdel(self):
            del self._latex
        return locals()
    latex = property(**latex())

    @property
    def coef(self):
        return self.__coef

    @coef.getter
    def coef(self):
        return self.__coef

    def _set_coef(self, value):
        self.__coef = value

    def Null(system):
        temp = Operator(system)
        temp.set_default(False)
        del temp.acting
        temp.struct = []
        temp._set_coef([])
        temp._label = 'O'
        return temp


    def loadlabel(system, label):
        temp = Operator.Null(system)
        #print("target : {}".format(label))
        for term in label.split('+'):
            temp1 = Operator(system)
            for op in term.split('@'):
                if op =='O' or len(op) == 0:continue
                if op[0] in Operator._table:
                    temp2 = Operator(system,prefix = Operator._table[op[0]])
                    if len(op[1:])>=2:
                        index = []
                        for ind in range(int(len(op[1:])/2)):
                            index.append(int(op[1+ind*2:3+ind*2]))
                        temp2 = temp2.index(*index)
                elif op[0] == 'U':
                    temp2 = Operator(system,prefix = system.get_function(int(op[1:])))
                temp1 = temp1@temp2
            temp += temp1
        return temp

    def save(self, force = False, saver = None):
        if self.name == None:  raise ValueError("This operator has no name. please run 'set_name()' first.")
        if saver is None:
            saver = self.system.saver.require_group('/operator')
        if saver.get(self.name, False) and not force:
            temp = saver[self.name][:]
            if (temp == self.__coef).all():
                pass
            else:
                saver[self.name] = self.__coef
        else:
            if saver.get(self.name,False):
                saver[self.name] = self.__coef
            else:
                saver.create_dataset(self.name, data = self.__coef)
            saver[self.name].attrs['label'] = self._label.encode()
            saver[self.name].attrs['name'] = self.name.encode()
            saver[self.name].attrs['group'] = False
        if self._latex:
            saver[self.name].attrs['latex'] = self._latex.encode()



        '''if os.path.isfile(self.system.path+'operator/'+self.name+'.xxz') and not force:
            temp = np.load(open(path+self.name+'.npy','rb'))
            if (temp == self.__coef).all():
                return
            else:
                np.save(open(path+self.name+'.npy','wb'),self.__coef)
        else:
            with open(path+self.name+'.xxz','w') as f:
                #pickle.dump(self.name,f,pickle.HIGHEST_PROTOCOL)
                f.write(self._label+'\n')
            np.save(open(path+self.name+'.npy','wb'),self.__coef,allow_pickle=True)'''

    def load(system, name, loader = None):
        if loader is None:
            if system.saver.is_exist('/operator/{}'.format(name)):
                loader = system.saver['/operator/{}'.format(name)]
                if loader.attrs.get('group',False):
                    return OperatorGroup.load(system, name)
            else:
                raise FileExistsError("Operator save file doesn't exist!")
        else:
            loader = loader[name]
        temp = Operator.loadlabel(system, loader.attrs['label'].decode())
        if loader.attrs.get('latex',False):
            temp.latex = loader.attrs['latex'].decode()
        temp._set_name(loader.name[loader.name.find('operator/')+9:])
        temp._set_coef(loader[:])
        return temp


    def __call__(self, target):
        if isinstance(target, State):
            temp = State(target.basis, (self.get_matrix(target.basis)@target.coef))
            return temp
        elif isinstance(target, np.ndarray):
            return self.acton(target)
        else:
            return self.acton([target])


    def set_default(self, bool):
        self.__default = bool

    def set_local(self, bool):
        self.__local = bool

    def is_identity(self):
        return self.__default

    def valid(self):
        return self.system.valid()

    def _set_name(self, name):
        if self.name in self.system._Operator: del self.system._Operator[self.name]
        self.name = name
        self.system._Operator[name] = self

    def set_name(self, name):
        #if name in self.system._op: raise KeyError("name : {}, Already Exist!.".format(name))
        if self.name in self.system._Operator: del self.system._Operator[self.name]
        self.name = name
        self.system._Operator[name] = self

    def set_group(self, group):
        if not self.name is None:
            #raise AttributeError("This operator has no name.")
            if self.name in self.system._Operator: del self.system._Operator[self.name]
        #self.group = True



    ############ matrix representation
    def acton(self, x): #x is array of state
        '''x is array of bitwise state(int type). return is another state(int type). default is Identity
        you can also implement of your own operator with keeping return format'''
        states = []
        x=np.array(x)
        for i,op in enumerate(self.struct):
            #op = np.vectorize(op)
            states.append((op(x), self.__coef[i]))
        results = []
        for _ in range(len(x)):
            results.append({})

        for st, coef in states:
            for j,s in enumerate(st):
                results[j][s] = results[j].get(s, 0) + coef
        return results


    def get_matrix(self, basis, eigen = False, keep = False,use_gpu= False):
        '''return matrix form of Operator as given basis'''
        '''if (self.__initial == basis.symmetry).all():
            if  not self.matrix is None: return self.matrix'''
        ############################# load must be implemented

        states = basis.state
        matsize = len(states)
        dtype = self.system.Odtype
        K = False
        if not basis.K == -1:
            K = True
        matrix = np.zeros([matsize**2],dtype = dtype)
        for i, result in enumerate(self.acton(states)):
            '''origin = np.array([st for st in result if not st<0])
            if len(origin)== 0 : continue

            value = np.array([result[st] for st in result if not st<0])
            target, coef = basis.convert(origin)

            mask = (target>=0)
            target = target[mask]
            coef = coef[mask]
            value = value[mask]
            period = basis.period(target)

            for st,c,v,p in zip(target,coef, value,period):
                matrix[st*matsize + i] += c*v*np.sqrt(basis.period(i)/p)'''

            origin = np.array(list(result.items()))
            if origin.shape[0]== 1 and origin[0][0] == -1: continue #continue only return is -1

            #value = np.array([result[st] for st in result if not st<0])
            target, coef = basis.convert(origin.T[0]) #convert key

            mask = (target>=0)
            #target = target[mask]
            #coef = coef[mask]
            #value = value[mask]
            period = basis.period(target)
            ip = basis.period(i)
            for M,st,c,p,O in zip(mask, target, coef,  period, origin):
                if not M: continue
                o, v  = O
                matrix[st*matsize + i] += c*v*np.sqrt(ip/p)


        if not K:
            matrix = matrix.real
        matrix = matrix.reshape(matsize,matsize)
        if eigen:
            C = self.system.Hamiltonian.get_eigenvectors(basis)
            if use_gpu:
                matrix_gpu = cp.asarray(matrix)
                B = cp.asarray(C)
                D = cp.asnumpy(B.conj().T@matrix_gpu@B)
                del matrix_gpu, B
                cp.get_default_memory_pool().free_all_blocks()
                return D
            return (C.conjugate().T@matrix@C).copy()
        return matrix.copy()


    def trace(self, basis_set, func = lambda x:x, eigen = True):
        '''return Tr{ f( O ) } on given sector(s). '''
        #eigen can be implemented
        if isinstance(basis_set, Basis):
            basis_set = [basis_set]
        length = 0
        vals = []
        for basis in basis_set:
            val = self.get_eigenvalue(basis)
            vals.append(val)
            length += len(val)
        func = np.vectorize(func)
        trace_val = np.empty([length],dtype = self.system.dtype)
        i = 0
        for val in vals:
            for v in val:
                trace_val[i] = v
                i+=1

        result = func(trace_val)
        #M = result.max()
        #m = result.min()
        #result = result.sort()
        #ratio = np.log10(M/m)
        #ratio /= 10
        return result.sum()

    def free(self):
        del self.matrix, self.__initial
        self.matrix = None
        self.__initial = False


    def expectation(self, state):
        r"""a funciton of getting expectation value of this operator based on given state.

        Parameters
        ----------
        state : ``xxzchain.core.State``
            state which will be calculated with.

        Returns
        -------
        ``np.float64`` or *(array)*
            array of :math:`\langle \psi | \hat O | \psi \rangle`

        """
        r'''
        return

        list of :math:`\langle \psi | O\hat | \psi \rangle`
        '''
        if len(state) == 1:
            return (state.dual()@ self.get_matrix(state.basis) @state.coef)
        else:
            return (state.dual()@ self.get_matrix(state.basis) @state.coef)[np.diag_indices(len(state))].copy()

    def act(self, state):
        return self.get_matrix(state.basis) @ state.coef.T

    def solve_on(self, basis, save = True):
        if self.name is None:
            print("For Solving eigenvalue problem, Operator must have its own name.")
            return
        print("Initializing.... target_basis = ({}) it may takes few minutes.\t\t\t".format(basis.symmetry),end='\r')

        if (basis.symmetry == (-1,-1,0,0)).all():
            saver = self.system.saver['basis'].require_group(self.name)
        else:
            saver = self.system.saver['basis/({},{},{},{})'.format(*basis.symmetry)].require_group(self.name)

        if not saver.get('eigenvector', False):
            val, vec = np.linalg.eigh(self.get_matrix(basis))
            if save:
                saver.create_dataset('eigenvector',data = vec)
                if not saver.get('eigenvalue', False):
                    saver.create_dataset('eigenvalue',data = val)
        else:
            val, vec = saver['eigenvalue'], saver['eigenvector']
        #self.__initial.append('basis/({},{},{},{})'.format(*basis.symmetry))
        return val, vec


    def get_eigenvalue(self, basis, save = True):
        '''return eigenvalues of matrix of this Operator on given basis.'''
        if not basis.data.get(self.name,False) or not basis.data[self.name].get('eigenvalue',False):
            val = np.linalg.eigvalsh(self.get_matrix(basis))
            saver = basis.data.require_group(self.name)
            if not saver.get('eigenvalue',False):
                if save:
                    saver.create_dataset('eigenvalue', data = val)
            return val



        return basis.data[self.name]['eigenvalue'][:]


    def get_eigenvectors(self, basis, save = True):
        '''return eigenvectors of matrix of this Operator on given basis.'''
        if not basis.data.get(self.name,False):
            #print("Initializing.... it may takes few minutes.")
            if not save:
                _, vec = self.solve_on(basis,save)
                return vec
            else:
                self.solve_on(basis,save)
        return basis.data[self.name]['eigenvector'][:]

    def get_eigenstates(self, basis, save = True):
        '''return State made up with eigenvectors of matrix of this Operator on given basis.'''
        return State(basis, self.get_eigenvectors(basis,save).T)



    ########## main operation   ##########

    def __mul__(self, other):
        if isinstance(other, self.__class__):raise OverflowError("Operator cannot be mul, please use @ symbol.")#if type(self)== type(other):
        temp = Operator(self.system)
        temp.set_default(self.__default)
        temp.struct = self.struct
        temp.acting = self.acting
        temp._label = self._label
        if type(other) == type(np.ufunc):
            self.__coef = other(self.__coef)
        else:
            temp._set_coef(self.__coef*other)
        return temp

    def __rmul__(self, other): return self.__mul__(other)

    def __add__(self, other):
        #print("me!")
        if isinstance(other, self.__class__):#if type(other) == type(self): #operator -> make parallel operation (merging)
            if self.__default and other.is_identity():       # dd Identity op -> calculate value sum
                return other.__add__(self.__coef)
            if not self.system == other.system: raise EnvironmentError("Operators cannot add between different system")
            temp = Operator(self.system)
            temp.set_default(False)
            struct = []
            label = ''
            l = len(self.struct)+len(other.struct)
            coef = np.zeros([l],dtype = self.system.Odtype)
            i= 0
            for la, co, op in zip(self._label.split('+'),self.__coef, self.struct):
                coef[i] = co
                struct.append(op)
                label+= la+'+'
                i+=1
            for la, co, op in zip(other._label.split("+"), other.__coef, other.struct):
                coef[i] = co
                struct.append(op)
                label+= la+'+'
                i+=1
            temp._set_coef(coef)
            temp.struct = struct
            temp._label = label[:-1]
            assert len(coef) == len(temp.struct),"{},{}".format(len(coef), len(temp.struct))
            return temp
        elif self.__default: #when self is identity -> calculate value(coefficient)
            temp = Operator(self.system)
            temp._set_coef(self.__coef+other)
            return temp
        else:
            temp = Operator(self.system)
            temp._set_coef(temp.coef*other)
            return self+temp
        raise ValueError("Value operation is only allowed for Identity.")


    def __radd__(self, other):
        #print("1")
        if not type(self) == type(other):
            return self.__add__(other)

    def __iadd__(self, other):
        #print("me!")
        if isinstance(other, self.__class__):#if type(other) == type(self): #operator -> make parallel operation (merging)
            if not self.system == other.system: raise EnvironmentError("Operators cannot add between different system")
            if self.__default and other.is_identity():       #Identity -> calculate value sum
                temp=Operator(self.system)
                temp._set_coef(self.__coef+other.coef)
                return temp
            l = len(self.coef)+len(other.coef)
            coef = np.zeros([l],dtype=self.system.Odtype)
            for i in range(len(self. coef)):
                coef[i] = self.coef[i]
            i = len(self.coef)
            for j in range(len(other.coef)):
                coef[i+j] = other.coef[j]
            label  = self._label
            struct = []
            for op in self.struct:
                struct.append(op)
            for la, op in zip(other._label.split('+'),other.struct):
                label += '+'+la
                struct.append(op)
            if label[0] == 'O':
                label = label[2:]
            temp = Operator(self.system)
            temp._set_coef(coef)
            if len(struct) == 1:
                temp.acting = struct[0]
            temp._label = label
            temp.struct = struct
            return temp
        else:
            self._set_coef(self.__coef+other)
            return self

    def __riadd__(self, other): return self.__iadd__(other)


    def __sub__(self, other):
        return self.__add__(-1*other)



    def __matmul__(self, other):
        #print("me!")
        if isinstance(other, self.__class__):#if type(other) == Operator:
            if not self.system == other.system: raise EnvironmentError("Operators cannot matmul between different system")

            if other.is_identity():
                return self * other.coef
            elif self.__default:
                return other * self.__coef
            temp = Operator(self.system)
            temp.set_default(False)
            struct = []
            label = ''
            l = len(other.struct)
            tl = len(self.struct) * l
            coef = np.zeros([tl],dtype = self.system.Odtype)
            def dec(a,b):
                def wrap(x):
                    return a(b(x))
                return wrap
            for i,val1 in enumerate(zip(self._label.split("+"),self.struct)):
                la1,my_op = val1
                for j,val2 in enumerate(zip(other._label.split("+"),other.struct)):
                    la2, o_op = val2
                    coef[i*l+j] = self.coef[i]*other.coef[j]
                    label += la1+'@'+la2 +'+'
                    func = np.vectorize(dec(my_op,o_op))
                    struct.append(func)
            temp._label = label[:-1]
            temp._set_coef(coef)
            temp.struct = struct
            return temp
        elif isinstance(other, State):
            if self == self.system.Hamiltonian:
                return self.get_eigenvalue(other.basis)*other.eigencoef
        else:
            raise ValueError("Only Operators matmul allowed.")



    ########## state calculation ##########
    #@np.vectorize                                # I
    def identity(x):
        return x

    #@np.vectorize                                # a
    def annihilation(x, i):
        if x == -1: return -1
        if (x>>i)&1:
            return x^(1<<i)
        return -1

    #@np.vectorize                                # c
    def creation(x, i):
        if x == -1: return -1
        if not (x>>i)&1:
            return x^(1<<i)
        return -1

    def pauli_x(x,i):
        return x^(1<<i)

    #@np.vectorize                                # n
    def n_i(x,i):
        if x == -1: return -1
        if (x>>i)&1:
            return x
        return -1

    #@np.vectorize                                # h
    def hopping(x,i,j):
        if x == -1: return -1
        m = (1<<i)|(1<<j)
        ch = m&x
        if  ch == m  or not ch: return -1
        return x^m


    def index(self, *arg):
        narg = []
        label = ''
        for a in arg:
            narg.append(a%self.system.size)
            label += "{}".format(a%self.system.size).zfill(2)
        temp = Operator(self.system)
        temp.set_default(False)
        lambda x: self.acting(x,*narg)
        def _index(x):
            return self.acting(x,*narg)
        temp.acting = np.vectorize(_index)
        temp._label = self._label+label
        temp.struct = [temp.acting]
        return temp


    x_i = pauli_x
    b_i = annihilation
    b_i_dag = creation

    _table = {'A':annihilation, 'C': creation, 'I' : identity, 'H':hopping, 'N':n_i, 'X':pauli_x}


# In[23]:


class State:
    path = '/state/'
    def __init__(self, basis, coef, eigen = False, name = None):
        '''state information get by basis, and this instance only contain its coefficient as vector ( shape of (,length) )'''
        self.system = basis.system
        self.basis = basis
        self.states = basis.state
        self.__coef,self.__eigencoef = None,None
        self.__e = None
        if eigen:
            self.eigencoef = coef.copy().T
        else:
            self.coef  = coef.copy().T
        self.time = 0
        self.length = self.coef.shape[1] if len(self.coef.shape) != 1 else 1
        #self.times = {}
        self.name = None
        if name is not None:
            self.set_name(name)
        #assert len(coef) == len(basis.state)

    @property
    def coef(self):
        return self.__coef

    @coef.setter
    def coef(self, coef):
        self.__coef = coef.astype(np.complex128)
        if len(self.__coef.shape) == 1:
            self.__coef = self.__coef.reshape(-1,1)
        if self.system.Hamiltonian is not None:
            self.__eigencoef = self.system.Hamiltonian.get_eigenvectors(self.basis).conjugate().T@self.__coef
        return

    @coef.getter
    def coef(self):
        return self.__coef

    @property
    def energy(self):
        return self.__e

    @energy.getter
    def energy(self):
        if self.system.Hamiltonian is None:
            raise EnvironmentError("Hamiltonian is None.")
        if self.__e is None:
            self.__e = self.system.Hamiltonian.expectation(self)
        return self.__e

    @property
    def eigencoef(self):
        return self.__eigencoef

    @eigencoef.setter
    def eigencoef(self, coef):
        if self.system.Hamiltonian is None:
            raise EnvironmentError("Hamiltonian is None.")
        self.__eigencoef = coef.astype(np.complex128)
        if len(self.__eigencoef.shape) == 1:
            self.__eigencoef = self.__eigencoef.reshape(-1,1)
        self.__coef = self.system.Hamiltonian.get_eigenvectors(self.basis)@self.__eigencoef
        return
    @eigencoef.getter
    def eigencoef(self):
        if self.system.Hamiltonian is None:
            raise EnvironmentError("Hamiltonian is None.")
        if self.__eigencoef is None:
            self.__eigencoef = self.system.Hamiltonian.get_eigenvectors(self.basis).conjugate().T@self.__coef
        return self.__eigencoef

    def norm(self):
        return np.linalg.norm(self.coef)

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        return State(self.basis, self.__coef.T[key].copy())

    def update(self):
        if self.system.Hamiltonian is None:
            raise EnvironmentError("Hamiltonian is None.")
        self.__eigencoef = self.system.Hamiltonian.get_eigenvectors(self.basis).conjugate().T@self.__coef
        self.__e = (self.system.Hamiltonian.get_eigenvalue(self.basis)*self.__eigencoef.T*self.__eigencoef.conjugate().T).sum(axis = 0)


    def set_name(self, name):
        if name in self.system._State: raise KeyError("'{}' is already exist!".format(name))
        if self.name is not None and self.name in self.system._State: del self.system._State[self.name]
        self.system._State[name] = self
        self.name  = name
        self.path = 'state/{}/'.format(self.name)

    def save(self, force = False):
        if self.name is None:
            raise NameError("This state has no name.")
        saver = self.system.saver.require_group(self.path)
        saver = saver.require_group(self.name)
        saver.attrs['basis'] = self.basis.symmetry
        saver.attrs['name'] = self.name
        if saver.get('coef',None) is None:
            saver.create_dataset('coef',data = self.__coef, compression = 'lzf')


    def load(system, name, force=None):
        loader = system.saver.get('state/'+name,None)
        if loader is None:
            print("'{}' State doesn't exist!".format(name))
            return
        symmetry = loader.attrs['basis']
        coef = loader['coef'][:]
        temp = State(system.get_basis(*symmetry),coef.T, name)
        return temp

    def dual(self):
        return self.coef.conjugate().T

    def stack(states):
        chenv = True
        length = 0
        basis = states[0].basis
        for state in states:
            chenv = chenv and state.basis == basis
            length += len(state)
        coef = np.zeros([length, len(basis.state)],dtype = states[0].system.dtype)
        for i, state in enumerate(states):
            coef[i] = state.coef.T
        return State(basis, coef)


    def __repr__(self):
        rep = ''
        rep += "'{}' state on sector {}\n".format(self.name, (self.basis.Q,self.basis.K))
        rep += "t             : {}\n".format(self.time)
        if self.__e is not None:
            rep += "Energy        : {}\n".format(self.__e)
        rep += 'State         : {}\n'.format(self.coef)
        if self.__eigencoef is not None:
            rep += '(eigen State) : {}\n'.format(self.eigencoef)
        return rep


    def time_evolving(self, t, vstack = False):
        if self.system.Hamiltonian is None:
            print("Hamiltonian is not defined. please define Hamiltonian of system.")
            return self
        H = self.system.Hamiltonian
        self.time = t
        self.__eigencoef *= np.exp(-1j*H.get_eigenvalue(self.basis)*t)
        self.__coef = H.get_eigenvectors(self.basis).T@self.eigencoef
        return self

    def time_evolving_states(self, t_range, H = None):
        if H is None:
            if  self.system.Hamiltonian is None:
                print("Hamiltonian is not defined. please define Hamiltonian of system.")
                return self
            H = self.system.Hamiltonian
        t = np.array(t_range)
        eigen = np.exp(-1j*H.get_eigenvalue(self.basis).reshape(-1,1)@t.reshape(1,-1))*self.eigencoef
        temp = State(self.basis, eigen.T, eigen=True)
        temp.time = t
        return temp


    def normalize(self, value = 1):
        self.coef *= value/np.linalg.norm(self.coef)


    def __add__(self, other):
        if not isinstance(other, self.__class__): raise ValueError("State only can be added with state.")
        if not self.basis == other.basis: raise EnvironmentError("Adding between different basis is not allowed")
        if len(self)  == 1:
            return other + self
        return State(self.basis, self.coef.T+other.coef.T)

    def __sub__(self, other):
        return self + -1*other

    def __mul__(self, other):
        return State(self.basis, self.coef.T*other)

    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            raise TypeError("matmul between states is not supported yet.")
        else:#if isinstance(other, Operator):
            raise SyntaxError("Did you mean State.T @ Operator?")




    def __rmatmul__(self, other):
        if isinstance(other, Operator):
            return State(self.basis, (other.get_matrix(self.basis)@self.coef).T)

    def entanglement_entropy_MSS(self, l):
        if l > self.system.size or l<0: raise IndexError
        elif l> self.system.size/2 : l = self.system.size - l
        fullstate = np.zeros([2**self.system.size, len(self)],dtype = self.system.Odtype)
        for i,state_coef in enumerate(self.__coef):
            fullstate[self.basis.state_set(self.basis.state[i])] = state_coef/self.basis.period(i)
        target = fullstate.T.reshape([len(self),-1,1<<l])
        _,s, _ = np.linalg.svd(target, full_matrices = True)
        ent = np.zeros([len(self)],dtype = np.float64)
        for i,line in enumerate(s):
            line = line[line!=0]
            sks = np.power(line, 2)
            ent[i] = (-sks*np.log(sks)).sum()
        return ent

    def print_states(self, full_state = False, state_set = False):
        return self.basis.print_states(full_state, state_set)




class Saver:
    def __init__(self,system):
        self.system = system
        self.__file = None
        self.__IO = None

    def open(self, path = None):
        '''create new hdf5 format data. if exist, open hdf5 '''
        if not self.__file is None:
            return
        if path is None:
            self.__file = h5py.File(self.system.path+self.system.name+'.hdf5','r+')
        else:
            self.__file = h5py.File(path, 'r+')

    def close(self):
        #del self.__file
        if not self.__file is None:
            self.__file.close()
            self.__file = None

    def set_file(self, h5obj):
        self.__file = h5obj

    def __del__(self):
        if self.__file is not None:
            self.__file.close()
        del self.__file


    def _create(self):
        #del self.__file
        self.__file = h5py.File(self.system.path+self.system.name+'.hdf5','a')
        self.__file.clear()
        #self.file.create_group('Workspace')

    def isfile(self, path = None):
        if path is None:
            return os.path.isfile(self.system.path+self.system.name+'.hdf5')
        else:
            return os.path.isfile(path)

    def is_exist(self, target_path):
        '''return True when Data exist on given path.'''
        ch  = not (self.file.get(target_path, None) is None)
        return ch

    def ls(self, path = None, tapping = '', show_all = False):
        if path is None:
            target = self.__file
            print(self.file.filename)
        else:
            target = self[path]
            print(tapping+target.name+" : dir")
        tapping += '\t'
        for name, item in list(target.items()):
            if isinstance(item, h5py.Group):
                if show_all:
                    self.ls(item.name, tapping, show_all)
                else:
                    print(tapping+item.name+" : dir")
            else:
                print(tapping+name)

    def initialize_basis(self, symmetry, **kwargs):
        """Initialize hdf5 save file on specific location.

        Parameters
        ----------
        symmetry : `list`
            basis symmtry. this will be a save location.

        kwargs
        ---------
            There are several things for saving.

        Return
        ---------
        `h5py.Group`
            The Group object of basis.
        """
        bsaver = self.file.require_group('/basis')
        folder = bsaver.require_group('({},{},{},{})'.format(*symmetry))
        if 'counts' in kwargs:
            folder.attrs['counts'] = kwargs['counts']
            del kwargs['counts']

        for i in kwargs:
            folder.create_dataset(i,data = kwargs[i],compression = 'lzf') # NOTE: address will be deprecated by optimizing
            #folder.create_dataset('address',data = np.array(list(address[n].items())),compression = 'lzf')
            #folder.create_dataset('period',data = np.array(period[n]),compression = 'lzf')
            #folder.attrs['counts'] = counts[n]
        #logger.debug("'{}' created, and 'state', 'period' added".format(symmetry))
        return folder

    @property
    def file(self):
        return
    @file.getter
    def file(self):
        if self.__file is None:
            self.open()
        return self.__file
    @property
    def attrs(self):  return self.open()
    @attrs.getter
    def attrs(self):
        return self.file.attrs

    #mapping original h5py File func.
    def require_group(self, *arg,**kwarg):  return self.file.require_group(*arg,**kwarg)
    def get(self, *arg,**kwarg):            return self.file.get(*arg,**kwarg)
    def flush(self):                        return self.file.flush()
    def items(self):                        return self.file.items()
    def keys(self):                         return self.file.keys()
    def values(self):                       return self.file.values()
    def __getitem__(self, key):             return self.file[key]
    def __delitem__(self, key):             del self.file[key]

    def __setitem__(self, key, value):
        if self.is_exist(key):
            self.file[key] = value
        else:
            self.files.attrs[key] = value



class Initializer:
    '''Basis sector calculator/'''
    comb = lambda self,q: int(math.factorial(self.size)/math.factorial(q)/math.factorial(self.size-q)) #nCq

    ## to check for 8-digits counting number of 1 table
    Qlist = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]     # set of 8 bit

    #translate 1110 => 1101 with size 4, translate state left side
    _translate = lambda self, n: (n>>(self.system.size-1))*((1)-(1<<self.system.size))+2*n
    #reverse translate 1110 => 0111 with size4, translate state right side
    _rtranslate = lambda self, n: ((n&1)<<self.system.size-1) + (n>>1)



    #spinz = lambda self, x,i: (x>>i)&1 # return 0, 1
    def parity(self, x):
        p=0
        for i in range(self.__l):
            p = p<<16
            p+= self.Plist[((1<<16)-1)&(x>>16*i)]
        p = p<<self.__r
        p += self.Plist[((1<<16)-1)&(x>>16*self.__l)]>>(16-self.__r)
        return p
    par_num = lambda self, x: self.Qlist[x&255]+self.Qlist[(x>>8)&255]+self.Qlist[(x>>16)&255]+self.Qlist[(x>>24)&255]
    @np.vectorize
    def _npar_num(x):
        return Initializer.Qlist[x&255]+Initializer.Qlist[(x>>8)&255]+Initializer.Qlist[(x>>16)&255]+Initializer.Qlist[(x>>24)&255]

    @np.vectorize
    def _ncheck(x, size):
        '''function to check whether given state is smallest state
        based on translational symmetry or not.
        return is period(positive integer), or error(-1)'''
        tr = lambda a, size:(a>>(size-1))*((1)-(1<<size))+2*a
        origin = x
        for i in range(size):
            a = tr(x,size)
            if a <origin:
                return -1
            elif a == origin:
                break
            x = a
        return i+1

    def binary(self, s):
        return bin(s)[2:].zfill(self.system.size)

    def __init__(self, system):
        '''Just need system,
        many method will take basis sector and give split sectors of them.'''
        self.system = system
        self.translate = np.vectorize(self._rtranslate)
        self.null = (-1,-1,0,0)
        self.init = [self.N, self.K, self.P,self.X]
        self.Plist = []
        self.__l, self.__r = int((self.system.size)/16), self.system.size%16
        for i in range(1<<16):
            p = 0
            for j in range(16):
                p+= ((i>>j)&1)<<(15-j)
            self.Plist.append(p)
        self.parity = np.vectorize(self.parity)
        self.logger = self.system.logger.getChild("Initializer")
        self.logger.debug("Initializer init.")

    def __del__(self):
        self.logger.debug("initializer deleted.")
        del self.logger

    def calculate(self, symmetry):
        symmetry =tuple(symmetry)
        self.logger.debug("basis calculation start.")
        for i in range(4):
            test = symmetry[:i+1] + self.null[i+1:]
            self.logger.debug("target : '{}', check : '{}'".format(symmetry, test))

            # test must have specific sector
            if test == self.null: continue
            if not self.system.saver.is_exist('/basis/({},{},{},{})'.format(*test)):
                self.logger.debug("basis ('{}') not found. basis initializer : '{}' started ".format(test, self.init[i]))
                self.init[i](self.system.get_basis(*(symmetry[:i] + self.null[i:])))  # take basis from prior basis

            if test == symmetry:
                self.logger.debug("calculation of target is complete. target : '{}'".format(symmetry))
                break
        '''if not self.system.saver.is_exist('/basis/({},{},{},{})'.format(symmetry[0], *self.null[1:])):
            self.Q(self.system.get_full_sector())
        elif not self.system.saver.is_exist('/basis/({},{},{},{})'.format(*symmetry[:2], *self.null[2:])):
            self.K(self.system.get_basis(symmetry[0]))'''
        #elif not self.system.saver.is_exist('/basis/({},{},{},{})'.format(symmetry[:3], *self.null[3:])):


    def N(self, basis):
        '''Initializer which find all small sector based on spin conservation.'''
        target = basis.state
        symmetry = basis.symmetry.copy()
        logger = self.logger.getChild("N")
        if (not symmetry[2]==0):
            raise EnvironmentError("Spin conserving and spin inversion are not commute on given basis.")
        logger.info("Initializing spin conserving sectors of given basis : ({},{},{},{})\t\t\t\t".format(*symmetry))

        #calculate basis sector
        state, address, period, counts = [],[],[],np.zeros([self.system.size+1],dtype = np.int64)
        for i in range(self.system.size+1):
            state.append([])
            address.append({})
            period.append([])
            counts[i] = 0
        N = self._npar_num(target)
        if symmetry[1] == -1:
            for n,s in zip(N,target):
                #logger.debug("\n\ttarget state : {},\n\tspin representation : {}, \n\tresult : {}".format(s, self.binary(s), n))
                state[n].append(s)
                address[n][s] = counts[n]
                counts[n]+=1
        else:
            raise SystemError("if this error occurs, then you must do debugging.")

        #save sectors
        bsaver = self.system.saver.file.require_group('/basis')
        for n in range(self.system.size+1):
            assert len(state[n]) == counts[n], 'component and its label unmatched'
            symmetry[0] = n
            self.system.saver.initialize_basis(symmetry, state = np.array(state[n]), period =np.array(period[n]), counts =counts[n] )
            #folder = bsaver.require_group('({},{},{},{})'.format(*symmetry))
            #folder.create_dataset('state',data = np.array(state[n]),compression = 'lzf') # NOTE: address will be deprecated by optimizing
            #folder.create_dataset('address',data = np.array(list(address[n].items())),compression = 'lzf')
            #folder.create_dataset('period',data = np.array(period[n]),compression = 'lzf')
            #folder.attrs['counts'] = counts[n]
            logger.debug("'{}' created, and 'state', 'period' added".format(symmetry))
        logger.info("Calculation ended for basis : {}".format(basis.symmetry.copy()))
        del logger






    def K(self, basis):
        '''Initializer which find all small sector based on translational invarient.'''
        #get infomation of target basis
        st = basis.state
        symmetry = basis.symmetry.copy()
        logger = self.logger.getChild('K')
        logger.info("Initializing momentum sectors of given basis : ({},{},{},{})\t\t\t\t".format(*symmetry))
        #make new empty storage
        state, address, period, distance, counts = [],[],{},{s:-1 for s in st},[]# [[] for q in range(self.size+1)],[[] for q in range(self.size+1)],[[] for q in range(self.size+1)],[[] for q in range(self.size+1)]
        states = {}
        for k in range(self.system.size):
            state.append([])
            address.append({})
            #period.append([])
            counts.append(0)


        #calculate period of each state based on traslation op
        pe = self._ncheck(st, self.system.size) ## calculate state if st is not rep. state return -1, else return period
        dist = {}
        St = st
        stpr = {}

        for s,p in zip(st,pe):
            stpr[s] = p  #get period from state
            if p<0:
                if not distance.get(s,False):
                    logger.critical("State('{}') which is not included traslational symmetry is founded.".format(s))
                continue
                print(s, p)
                address[k][s] = address[k][self._rtranslate(s)] #acsending order. so, the bigger one comes later.

            #debug
            elif p ==0:
                raise ValueError("something wrong with state ({}) got period 0.".format(s))
            else:
                #make representative state that is consist of states.
                logger.debug("State '{}' is selected as representative state(RS) with period '{}'.".format(s, p))
                states[s] = [s]                  #state_set (set of states which matched with translational symmetry)
                period[s] = p
                ts = s
                distance[s] = 0
                for l in range(p-1):
                    ts = self._rtranslate(ts) #bit shift to left periodically
                    logger.debug("\tT^{:>2} |{}> : {}  ({:>2}).".format(l+1, self.binary(s), self.binary(ts),ts))
                    states[s].append(ts)
                    distance[ts] = l+1
                logger.debug("All members of RS '{}' is founded.")
            logger.debug("And RS '{}' is belonging to".format(s))
            for k in range(self.system.size):
                if k*p % self.system.size == 0:
                    state[k].append(s)
                    logger.debug("\tk = {:>2} momentum sector.".format(k, s, p))
                    #address[k][s] = counts[k]
                    #period[k].append(p)
                    counts[k] += 1
        l = len(state[0])
        kstate = -1*np.ones([l,self.system.size+1],dtype = np.int64)
        i=0
        for key, val in list(states.items()):
            kstate[i,0] = key
            assert len(val) == period[key]
            for j,value in enumerate(val):
                kstate[i,j+1] = value
            i+=1

        for key in distance:
            if distance[key] == -1:
                logger.critical("State '' still has negative distance(assertion).")
                assert distance[key] != -1, "state '{}' is not considered.".format(key)


        #deprecated code
        '''dist = {}
        adrs = {}
        state_to_period = lambda x: stpr.get(x)
        state_to_period = np.vectorize(state_to_period)
        for d in range(self.system.size):
            ch = state_to_period(St)
            for chs,chks in zip(st[np.logical_not(ch==-1)],St[np.logical_not(ch==-1)]):
                dist[chs] = d
                adrs[chs] = address[0][chks]
                assert d<state_to_period(chks),"d {},pe {}".format(d, state_to_period(chks))
            St = St[ch ==-1]
            if len(St) == 0 : break
            st = st[ch ==-1]
            St = self.translate(St)

        for k in range(self.system.size):
            for s in state[k]:
                if address[k].get(s,None) is None:
                    address[k][s] = adrs[s]
                    distance[k][s] = dist[s]'''

        bsaver = self.system.saver.file.require_group('/basis')

        for k in range(self.system.size):
            assert len(state[k]) == counts[k], 'component and its label unmatched'
            symmetry[1] = k
            self.system.saver.initialize_basis(symmetry, counts=counts[k], state=np.array(state[k]))
            #folder = bsaver.require_group('({},{},{},{})'.format(*symmetry))
            #folder.create_dataset('state',data = np.array(state[k]),compression = 'lzf')
            #folder.create_dataset('address',data = np.array(list(address[k].items())),compression = 'lzf')

            logger.debug("'{}' created, and 'state', 'period', 'distance', 'state_set' are added".format(symmetry))
            if k ==0 :
                folder = self.system.saver.initialize_basis(symmetry, counts=counts[k], state=np.array(state[k]))
                self.system.saver(symmetry, period=np.array(list(period.items())),distance=np.array(list(distance.items())), state_set=kstate)
                #folder.create_dataset('period',data = np.array(list(period.items())),compression = 'lzf')
                #folder.create_dataset('distance',data = np.array(list(distance.items())),compression = 'lzf')
                #folder.create_dataset('state_set',data = kstate,compression = 'lzf')
                ks = folder['state_set']
                ds = folder['distance']
                peri = folder['period']
            else:
                folder['state_set'] = ks
                folder['distance'] = ds
                folder['period'] = peri
            #folder.attrs['counts'] = counts[k]
        logger.info("Calculation ended for basis : {}".format(basis.symmetry.copy()))
        del logger

    def X(self, basis):
        '''initializer for spin inversion symmetry'''
        full = self.system.max - 1
        target = basis.state
        symmetry = basis.symmetry.copy()
        #origin_address = basis.address
        logger = self.logger.getChild("X")
        if not (symmetry[0] == self.system.size/2 or symmetry[0]==-1): raise EnvironmentError("Spin conserving and spin inversion are not commute.")
        logger.info("Initializing spin inversion sectors of given basis : ({},{},{},{})\t\t\t\t".format(*symmetry))

        #calculate basis sector
        s_state, s_address, s_period, s_counts = [],{},{},0
        d_state, d_address, d_period, d_counts = [],{},{},0
        state_set = {}
        d_bar = []
        '''for i in range(2):
            state.append([])
            address.append({})
            period.append([])
            counts[i] = 0'''

        F = target^full

        # WARNING:  must be check for period 4 in x and p
        if symmetry[1] == -1: # no momentum representative sector
            for f,s in zip(F,target):
                if s<f:
                    logger.debug("target : {}, flip : {}, selected doublet.".format(s,f))
                    d_state.append(s)
                    d_address[s] = d_counts
                    d_counts +=1
                    state_set[s] = [s,f]
                    #d_period.append(2)
                elif s == f:
                    logger.debug("target : {}, flip : {}, singlet.".format(s,f))
                    s_state.append(s)
                    s_address[s] = s_counts
                    s_counts +=1
                    #s_period.append(1)
                else:
                    logger.debug("target : {}, flip : {}, rejected doublet.".format(s,f))
                    d_address[s] = d_address[f] #if f is not in d_address occur error
                    #check representative which is not selected = d_bar
                    d_bar.append(s)
            s_period = {s_state[i] : 1 for i in range(len(s_state))}#np.ones([len(s_state)], dtype = np.int32)
            d_period = {d_state[i] : 2 for i in range(len(d_state))}#2*np.ones([len(d_state)], dtype = np.int32)
        elif symmetry[1] == 0:
            distance = {}
            for f,s in zip(F,target):
                #find representative of result
                fs = basis.state[basis.find(f)]
                if s == fs:
                    logger.debug("target : {}, flip : {}, singlet.".format(s,f))
                    s_state.append(s)
                    for comp in basis.state_set(s):
                        s_address[comp] = s_counts
                        distance[comp] = basis.distance(comp)
                    s_counts +=1
                    state_set[s] = basis.state_set(s)
                    s_period[s] = basis.period(basis.find(s))
                elif s<fs:
                    logger.debug("target : {}, flip : {}, selected doublet.".format(s,f))
                    d_state.append(s)
                    state_set[s] = list(basis.state_set(s))
                    for comp in basis.state_set(s):
                        d_address[comp] = d_counts
                        distance[comp] = basis.distance(comp)
                    d_counts +=1
                    d_period[s] = basis.period(basis.find(s))+basis.period(basis.find(fs))
                else:
                    logger.debug("target : {}, flip : {}, rejected doublet.".format(s,f))
                    for comp in basis.state_set(s):
                        d_address[comp] = d_address[fs]
                        distance[comp] = basis.distance(comp)
                        state_set[fs].append(comp)
                        d_bar.append(comp)
        assert len(s_state) == s_counts
        assert len(d_state) == d_counts
        states = -1*np.ones([s_counts+d_counts,self.system.size*4+1],np.int32)
        for i,key in enumerate(state_set):
            logger.debug("Family of '{}' :".format(key))
            states[i,0] = key
            for j,value in enumerate(state_set[key]):
                logger.debug("\t'{}',".format(value))
                states[i,j+1] = value
            logger.debug("Total {} states Done.".format(j+1))

        for key in d_address:
            s_address[key] = d_address[key]+s_counts

        period = {}
        for s in s_period:
            period[s] = s_period[s]
        for s in d_period:
            period[s] = d_period[s]
        #save sectors
        bsaver = self.system.saver.file.require_group('/basis')
        symmetry[3] = 1
        folder = self.system.saver.initialize_basis(symmetry,
                                    state=np.array(s_state+d_state),
                                    address=np.array(list(s_address.items())),
                                    period = np.array(list(period.items())),
                                    state_set= states,
                                    counts =s_counts + d_counts
                                    )
        #folder = bsaver.require_group('({},{},{},{})'.format(*symmetry))
        symmetry[3] = -1
        folder_ = self.system.saver.initialize_basis(symmetry,
                                    state=np.array(d_state),
                                    address=np.array(list(d_address.items())),
                                    Xd_bar=np.array(d_bar),
                                    counts=d_counts
                                    )

        folder_['period'],folder_['state_set'] = folder['period'],folder['state_set']
        #folder_ = bsaver.require_group('({},{},{},{})'.format(*symmetry))
        #folder.create_dataset('state',data = np.array(s_state+d_state),compression = 'lzf')
        #folder_.create_dataset('state',data = np.array(d_state),compression = 'lzf')

        #folder.create_dataset('address',data = np.array(list(s_address.items())),compression = 'lzf')
        #folder_.create_dataset('address',data = np.array(list(d_address.items())),compression = 'lzf')
        #folder.create_dataset('period',data = np.array(list(period.items())),compression = 'lzf')
        #folder_['period'] = folder['period']
        #folder.create_dataset('period',data = np.array(s_period+d_period),compression = 'lzf')
        #folder_.create_dataset('period',data = np.array(d_period),compression = 'lzf')
        #folder.create_dataset('state_set',data = states, compression = 'lzf')
        #folder_['state_set'] = folder['state_set']

        if symmetry[2]==-1:
            folder_['Pd_bar'] = bsaver['({},{},{},{})/Pd_bar'.format(symmetry[0],symmetry[1],-1,0)]
            folder['Pd_bar'] = bsaver['({},{},{},{})/Pd_bar'.format(symmetry[0],symmetry[1],-1,0)]
        folder_.create_dataset('Xd_bar',data = np.array(d_bar),compression = 'lzf')
        if symmetry[1] == 0:
            folder.create_dataset('distance',data = np.array(list(distance.items())),compression = 'lzf')
            folder_['distance'] = folder['distance']

        #folder.attrs['counts'] = s_counts + d_counts
        #folder_.attrs['counts'] = d_counts

    def P(self, basis):
        '''initializer for parity symmetry(same methodology with F)'''
        full = self.system.max - 1
        target = basis.state
        symmetry = basis.symmetry.copy()
        logger = self.logger.getChild("P")
        #origin_address = basis.data['address']
        if not (symmetry[1] == self.system.size/2 or symmetry[1]==-1 or symmetry[1]==0): raise EnvironmentError("Translation and parity are not commute.")
        logger.info("Initializing parity sectors of given basis : ({},{},{},{})\t\t\t\t".format(*symmetry))
        #calculate basis sector
        s_state, s_address, s_period, s_counts = [],{},{},0
        d_state, d_address, d_period, d_counts = [],{},{},0
        state_set = {}
        d_bar = []
        '''for i in range(2):
            state.append([])
            address.append({})
            period.append([])
            counts[i] = 0'''

        F = self.parity(target)


        if symmetry[1] == -1:
            for f,s in zip(F,target):
                if s<f:
                    d_state.append(s)
                    d_address[s] = d_counts
                    d_counts +=1
                    d_period[s] = 2
                elif s == f:
                    s_state.append(s)
                    s_address[s] = s_counts
                    s_counts +=1
                    s_period[s] = 1
                else:
                    d_address[s] = d_address[f]
                    d_bar.append(s)
        elif symmetry[1] == 0:
            distance = {}
            for f,s in zip(F,target):
                fs = basis.state[basis.find(f)]
                if s == fs:
                    logger.debug("target : {}, flip : {}({}), singlet.".format(s,f,fs))
                    s_state.append(s)
                    for comp in basis.state_set(s):
                        s_address[comp] = s_counts
                        distance[comp] = basis.distance(comp)
                    s_counts +=1
                    state_set[s] = basis.state_set(s)
                    s_period[s] = (basis.period(basis.find(s)))
                elif s<fs:
                    logger.debug("target : {}, flip : {}, selected doublet.".format(s,f))
                    d_state.append(s)
                    state_set[s] = list(basis.state_set(s))
                    for comp in basis.state_set(s):
                        d_address[comp] = d_counts
                        distance[comp] = basis.distance(comp)
                    d_counts +=1
                    d_period[s] = (basis.period(basis.find(s))+basis.period(basis.find(fs)))
                else:
                    logger.debug("target : {}, flip : {}, rejected doublet.".format(s,f))
                    for comp in basis.state_set(s):
                        d_address[comp] = d_address[fs]
                        distance[comp] = basis.distance(comp)
                        state_set[fs].append(comp)
                        d_bar.append(comp)
        assert len(s_state) == s_counts
        assert len(d_state) == d_counts
        states = -1*np.ones([s_counts+d_counts,self.system.size*2+1],np.int32) #max period is 4L L from k, 2 from x and r respectively
        for i,key in enumerate(state_set):
            states[i,0] = key
            for j,value in enumerate(state_set[key]):
                states[i,j+1] = value

        for key in d_address:
            s_address[key] = d_address[key]+s_counts

        period = {}
        for s in s_period:
            period[s] = s_period[s]
        for s in d_period:
            period[s] = d_period[s]

        #save sectors
        symmetry[2] = 1
        folder = self.system.saver.initialize_basis(symmetry,
                                            state=np.array(s_state+d_state),
                                            address=np.array(list(s_address.items())),
                                            period=np.array(list(period.items())),
                                            state_set=states,
                                            counts=s_counts + d_counts
                                            )
        #bsaver = self.system.saver.file.require_group('/basis')
        #folder = bsaver.require_group('({},{},{},{})'.format(*symmetry))
        symmetry[2] = -1
        folder_ = self.system.saver.initialize_basis(symmetry,
                                            state=np.array(d_state),
                                            address=np.array(list(d_address.items())),
                                            Pd_bar=np.array(d_bar),
                                            counts=d_counts
                                            )
        #folder_ = bsaver.require_group('({},{},{},{})'.format(*symmetry))
        #folder.create_dataset('state',data = np.array(s_state+d_state),compression = 'lzf')
        #folder_.create_dataset('state',data = np.array(d_state),compression = 'lzf')

        #folder.create_dataset('address',data = np.array(list(s_address.items())),compression = 'lzf')
        #folder_.create_dataset('address',data = np.array(list(d_address.items())),compression = 'lzf')

        #folder.create_dataset('period',data = np.array(list(period.items())),compression = 'lzf')
        folder_['period'] = folder['period']
        #folder.create_dataset('period',data = np.array(s_period+d_period),compression = 'lzf')
        #folder_.create_dataset('period',data = np.array(d_period),compression = 'lzf')
        #folder_.create_dataset('Pd_bar',data = np.array(d_bar),compression = 'lzf')
        #folder.create_dataset('state_set',data = states, compression = 'lzf')
        if symmetry[1] == 0:
            folder.create_dataset('distance',data = np.array(list(distance.items())),compression = 'lzf')
            folder_['distance'] = folder['distance']
            folder_['state_set'] = folder['state_set']
        #folder.attrs['counts'] = s_counts + d_counts
        #folder_.attrs['counts'] = d_counts



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
        for op in self._index:
            self._operators[self._index[op]].save(force = force, saver = gsaver)
            gsaver[self._operators[self._index[op]].name].attrs['index'] = op
            gsaver[self._operators[self._index[op]].name].attrs['key'] = self._index[op].encode()
        gsaver.attrs['des'] = str(self.description).encode()
        gsaver.attrs['i'] = self._i
        gsaver.attrs['group'] = True

    def __len__(self):
        return len(self._operators)

    def __iter__(self):
        for i in range(self._i):
            yield self._operators[self._index[i]]


    def set_name(self, name):
        #if name in self.system._op: raise KeyError("name : {}, Already Exist!.".format(name))
        if self.name in self.system._Operator: del self.system._Operator[self.name]
        self.name = name
        self.system._Operator[name] = self


    def load(system, name):
        if system.saver.is_exist('/operator/{}'.format(name)):
            temp = OperatorGroup(system, name)
            loader = system.saver['/operator/{}'.format(name)]
            i = 0
            for op in loader:
                temp[loader[op].attrs['key'].decode()] = Operator.load(system, op, loader)

                i+=1
            for op in loader:
                temp._index[loader[op].attrs['index']] = loader[op].attrs['key'].decode()
            assert i == loader.attrs['i']
            temp._i = i
        return temp

    def __setitem__(self, key, value):
        '''make operator as member of group '''
        #if str(key) in self._operators:
            #self._operators[str(key)] = value
            #raise KeyError("key already exist! if you want to delete, use 'del ogj[key]'.")
        if isinstance(value, Operator):

            if not str(key) in self._operators:
                self._index[self._i] = str(key)
                self._i += 1
            self._operators[str(key)] = value
            value.set_name(self.name+"."+str(key))
            value.set_group(self)

        else:
            raise ValueError("Only Operator can be grouped")
        return

    def __delitem__(self, key):
        target = None
        if isinstance(key, int):
            target = key
        else:
            for i in self._index:
                if self._index[i] == key:
                    target = i
                    break
        if target is None:
            raise KeyError(str(key))
        for i in range(target+1, self._i):
            self._index[i] = self._index[i+1]
        del self._index[self._i], self._operator[key]
        self._i -=1


    def __getitem__(self, key):
        if key in self._operators:
            return self._operators[key]
        elif isinstance(key, int):
            return self._operators[self._index[key]]
        else:
            raise NotImplemented()


    def __call__(self, key):
        return self.__getitem__(key)
