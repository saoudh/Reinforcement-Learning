import numpy as np
from gym.spaces import Box


class RLDataset:
    """
    RL dataset is a class for storing sample of shape <s,a,r,s',t> where
    s = current state
    a = action used
    r = reward
    s' = next state
    t = 1 if the state is NOT terminal, 0 else
    """
    def __init__(self, s, a, r, s_next, t, validation=0.2, batch_size=100):
        elements = {'s':s, 'a':a, 'r':r, 's_next':s_next, 't':t}

        #check that all the column vector have same first dimension
        for (x,y) in zip(list(elements.values())[1:],list(elements.values())[:-1]):
            self.N = x.shape[0]
            if not x.shape[0] == y.shape[0]:
                raise Exception("s, a, r, s_next should have all the same first dimension")

        #check that reward is a (N x 1) column vector
        if not r.shape[1]==1:
            raise Exception("r.shape[1] != 1")

        # check that terminate is a (N x 1) column vector
        if not t.shape[1] == 1:
            raise Exception("t.shape[1] != 1")

        #check that the state and next shape have same dimension
        if not s.shape[1] == s_next.shape[1]:
            raise Exception("s and s_next should have exactly same dimensions")

        self.elements = elements
        self.validation = validation
        self.N_train = int(self.N * (1-validation))
        self.N_validation = self.N - self.N_train
        self.mode = 'train'
        self.batch_indx = range(self.N)
        np.random.shuffle(self.batch_indx)
        self.i_batch = 0
        self.batch_size = batch_size

    @property
    def s(self):
        return self.elements['s'][self.get_indx(),:]

    @property
    def a(self):
        return self.elements['a'][self.get_indx(),:]

    @property
    def r(self):
        return self.elements['r'][self.get_indx(),:]

    @property
    def s_next(self):
        return self.elements['s_next'][self.get_indx(),:]

    @property
    def t(self):
        return self.elements['t'][self.get_indx(),:]

    def set_mode(self, mode):
        self.mode = mode

    def get_indx(self):
        if self.mode=='train':
            return np.array(range(self.N_train))
        elif self.mode=='validation':
            return self.batch_indx[self.N_train:]
        elif self.mode=='batch':
            min_i = self.i_batch * self.batch_size
            max_i = min([(self.i_batch + 1) * self.batch_size, self.N_train - 1])
            return self.batch_indx[min_i:max_i]
        elif self.mode=='all':
            return self.batch_indx[:]
        else:
            raise Exception("Mode " + self.mode + " not existing. Choose between train | validation | batch")

    def next_batch(self):
        self.i_batch += 1
        if self.s.shape[0] < self.batch_size:
            self.i_batch = 0
            self.shuffle()
            return False, self.s, self.a, self.r, self.s_next, self.t
        if self.i_batch*self.batch_size  > self.N_train -3:
            self.i_batch = 0
            self.shuffle()
            return False, self.s, self.a, self.r, self.s_next, self.t
        return True, self.s, self.a, self.r, self.s_next, self.t

    def shuffle(self):
        np.random.shuffle(self.batch_indx)

    def reset_pointer(self):
        self.i_batch = 0


class RLBuffer(RLDataset):
    """
    RLBuffer is an extension of RLDataset.
    Whenever a new sample is added to the Buffer,
        1) the oldest sample is rejected if the dataset has reached the maximum size
        2) the indexes of the dataset are shuffled
    """
    def __init__(self, environment, validation=0, batch_size=100, max_size=1000):

        state_dim = environment.observation_space.high.shape[0]
        action_dim = 1
        if type(environment.action_space) is Box:
            action_dim = environment.action_space.high.shape[0]

        self.state_dim = state_dim
        self.action_dim = action_dim
        RLDataset.__init__(self, np.zeros((0, state_dim)),
                    np.zeros((0, action_dim)),
                    np.zeros((0,1)),
                    np.zeros((0,state_dim)),
                    np.zeros((0,1)), validation, batch_size)
        self.max_size = max_size
        self.N = self.elements['s'].shape[0]
        self.N_train = int(self.N * (1 - self.validation))
        self.N_validation = self.N - self.N_train
        self.batch_indx = range(self.N)
        np.random.shuffle(self.batch_indx)

    def add_sample(self,s,a,r,s_next,t ):
        self.push('s',s)
        self.push('a',a)
        self.push('r',r)
        self.push('s_next', s_next)
        self.push('t',t)

        self.N = self.elements['s'].shape[0]

        #if not np.array_equal(self.elements['s'][-1:,:], s):
        #    print "errors"

        self.N_train = int(self.N * (1 - self.validation))
        self.N_validation = self.N - self.N_train
        self.batch_indx = list(range(self.N))
        np.random.shuffle(self.batch_indx)

    def push(self, stack_name, element):
        n_col = self.elements[stack_name].shape[1]
        well_shaped_element = np.reshape(element, (1,n_col))
        self.elements[stack_name] =\
            np.append(self.elements[stack_name]
            ,well_shaped_element, axis=0)[-self.max_size:,:]

    def reset(self):
        self.elements = {'s':np.zeros((0, self.state_dim)),
                         'a':np.zeros((0, self.action_dim)),
                         'r':np.zeros((0,1)),
                         's_next':np.zeros((0,self.state_dim)),
                         't':np.zeros((0,1))}