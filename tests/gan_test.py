import numpy as np
import theano
import theano.tensor as T
import argparse
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from collections import OrderedDict

sns.set(color_codes=True)
seed = 11
np.random.seed(seed)

def disciminator_loss(D1,D2):
    return T.mean(-(T.log(D1) + T.log(1-D2)))

def generator_loss(D2):
    return T.mean(-T.log(D2))

def nll(y,p_y_given_x):
    return(-T.sum(T.log(p_y_given_x)[T.arange(y.shape[0]), y]))

def cce(y,p_y_given_x):
    return T.nnet.categorical_crossentropy(p_y_given_x,y).mean()

def bce(y, p_y_given_x):
    return T.nnet.binary_crossentropy(p_y_given_x,y).mean()

def mse(y, p_y_given_x):
    return ((y-p_y_given_x)**2).mean()

def absolute_error(y, p_y_given_x):
    return T.abs_(y-p_y_given_x).mean()

class DataDistribution:

    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self,N):
        samples = np.linspace(self.mu,self.sigma,N)
        samples.sort()
        return samples

    def gaussian_likelihood(self,X):
        return (1. / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-(((X - self.mu) ** 2) / (2 * self.sigma ** 2)))

class GenerationDistribution:

    def __init__(self,range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01

class Generator(object):

    def __init__(self,X,hidden_size):
        self.w,self.b = norm_weights(X.get_value(borrow=True, return_internal_type=True).shape[0], hidden_size)
        self.X = X
        self.hidden_size = 2

    def generator(self):
        h0 = T.nnet.softplus(T.dot(self.X,self.w)+self.b)
        h1 = T.nnet.relu(h0,0)
        return h1

def generator(X,w,b):
    h0 = T.nnet.softplus(T.dot(X, w) + b)
    h1 = T.nnet.relu(h0, 0)
    return h1

class Discriminator(object):

    '''
        if minibatch_layer:
            h2 = minibatch(h1)
            print("Minibatch h2 shape : ",h2.get_shape())
        else:
            h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))
            print("Normal h2 shape : ", h2.get_shape())

        h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
        print("output shape ", h3.get_shape())
    '''

    def __init__(self,X, n_in, n_hidden, n_out,minibatches=False):

        self.X = X
        self.n_hidden = n_hidden
        self.n_in = n_in
        self.n_out = n_out
        self.minibatches = minibatches
        self.h = []
        self.get_disc_rnn_weights()

    def get_disc_rnn_weights(self):
        # recurrent weights as a shared variable
        W_init = np.asarray(np.random.uniform(size=(self.n_hidden, self.n_hidden),
                                              low=-.01, high=.01),
                            dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name='W')
        # input to hidden layer weights
        W_in_init = np.asarray(np.random.uniform(size=(self.n_in, self.n_hidden),
                                                 low=-.01, high=.01),
                               dtype=theano.config.floatX)
        self.W_in = theano.shared(value=W_in_init, name='W_in')

        # hidden to output layer weights
        W_out_init = np.asarray(np.random.uniform(size=(self.n_hidden, self.n_out),
                                                  low=-.01, high=.01),
                                dtype=theano.config.floatX)
        self.W_out = theano.shared(value=W_out_init, name='W_out')

        h0_init = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
        self.h0 = theano.shared(value=h0_init, name='h0')

        bh_init = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init, name='bh')

        by_init = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.by = theano.shared(value=by_init, name='by')

        self.params = [self.W, self.W_in, self.W_out, self.h0,
                       self.bh, self.by]

    def get_weights(self):

        # W = scale * np.random.randn(nin, nout)
        w1,b1 = norm_weights(self.n_hidden*2,self.X.get_value(borrow=True, return_internal_type=True).shape[1])
        w2, b2 = norm_weights(self.n_hidden*2,self.n_hidden*2)
        w3, b3 = norm_weights(1,self.n_hidden*2)
        self.w = {'w_in':w1,'w_h':w2, 'w_out':w3}
        self.b = {'b1':b1, 'b2':b2, 'b3': b3}

        return [self.w, self.b]

    def discriminator(self):

        for i in range(self.n_hidden):

            if i==0:
                g_z = T.dot(self.X,self.w[i])
                self.h[i] = T.nnet.softplus(g_z)
            elif i==2 and self.minibatches:
                self.h[i] = T.nnet.softplus(minibatch(self.n_hidden))
            elif i < self.n_hidden:
                self.h[i] = T.nnet.softplus(T.dot(self.h[i], self.w[i]) + self.b[i])
            else:
                self.h[i] = T.nnet.relu(self.h[i], 0)

        return [self.h, self.w, self.b]

    def rnn_discriminator(self):
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)

        # recurrent function (using tanh activation function) and linear output
        # activation function
        def step(x_t, h_tm1):
            h_t = T.nnet.softplus(T.dot(x_t, self.W_in) + \
                                  T.dot(h_tm1, self.W) + self.bh)
            y_t = T.dot(h_t, self.W_out) + self.by
            return h_t, y_t

        # the hidden state `h` for the entire sequence, and the output for the
        # entire sequence `y` (first dimension is always time)
        [self.h, self.y_pred], _ = theano.scan(step,
                                               sequences=self.X,
                                               outputs_info=[self.h0, None])
        return (self.h,self.y_pred)

    def updater(self, loss, params, input, learning_rate=0.01):
        decay = 0.95
        num_decay_steps = 150
        batch = theano.shared(0)

        dW = T.grad(loss, params['W'])
        dW_in = T.grad(loss, params['W_in'])
        dW_out = T.grad(loss, params['W_out'])

        gradients = theano.function(inputs=input, outputs=loss, updates=
        OrderedDict([(params['W'], params['W'] - learning_rate * dW),
                     (params['W_in'], params['W_in'] - learning_rate * dW_in),
                     (params['W_out'], params['W_out'] - learning_rate * dW_out)]), on_unused_input='ignore')
        return gradients

    '''
    def discriminator(input, hidden_size, minibatch_layer=True):
        inc = 0
        h = []
        for i in range(hidden_size):
            if i == 0:
                h[i] = T.nnet.softplus(T.dot(self.X, self.w[i]) + self.b[i], name='h' + str(i))
            else:
                h[i] = T.nnet.softplus(T.dot(self.h[i], self.w[i]) + self.b[i], name='h' + str(i))
            inc += 1
        h[inc] = T.nnet.relu(h[inc], 0)
        return [self.h, self.w, self.b]
    '''

    '''
    self.W = theano.shared(
    value=np.zeros(
        (self.batch_size, 1),
        dtype=theano.config.floatX
    ),
    name='W_pre',
    borrow=True
)
# initialize the biases b as a vector of n_out 0s
self.b = theano.shared(
    value=np.zeros(
        (self.batch_size, 1),
        dtype=theano.config.floatX
    ),
    name='b_pre',
    borrow=True
        )

    '''


'''
id : 1

W_disc = OrderedDict()
b_disc = OrderedDict()

for i in range(self.mlp_hidden_size):
    if i == 0:
        W_disc[i],b_disc[i] = norm_weights(X.shape[0], h_dim * 2)
    elif (i>0) and (i<len(self.mlp_hidden_size)):
        W_disc[i],b_disc[i] = norm_weights(h_dim * 2, h_dim * 2)
    else:
        W_disc[i],b_disc[i] = norm_weights(h_dim * 2, 1)

self.W = W_disc
self.b = b_disc
'''

def get_synthetic(n_hidden=10,n_in=5, n_out=2,n_steps=10, n_seq=100):
    '''
    Test RNN with real-valued outputs.
    n_hidden = 10
    n_in = 5
    n_out = 2
    # number of recurrent steps
    n_steps = 10
    # number of training sentences
    n_seq = 100
    '''

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_steps, n_in)
    targets = np.zeros((n_seq, n_steps, n_out))

    targets[:, 1:, 0] = seq[:, :-1, 3]  # delayed 1
    targets[:, 1:, 1] = seq[:, :-1, 2]  # delayed 1
    targets[:, 2:, 2] = seq[:, :-2, 0]  # delayed 2

    targets += 0.01 * np.random.standard_normal(targets.shape)

    return(seq,targets)

def norm_weights(fan_out,fan_in=None,layer_num=1,name='',bias=True,std=1.0):

    print("Fan in : ",fan_in)
    print("Fan out : ", fan_out)

    if bias:
        std= 1.0/np.sqrt(std)

        w = theano.shared(name=name,
                          value=0.1 * np.random.uniform(-std, std,(fan_in,fan_out)).astype(theano.config.floatX))
        b = theano.shared(name=name,
                          value=0.1 * np.random.uniform(-std, std, (fan_out,)).astype(theano.config.floatX))
        return [w,b]
    else:
        return theano.shared(name=name+str(layer_num), value=0.1 * np.random.uniform(-std, std,(fan_in,fan_out)).astype(theano.config.floatX))

def ortho_weight(ndim):

    W = np.random.randn(ndim,ndim)
    s, u, v = np.linalg.svd(W)
    return v

# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')

def linear(input, output_dim, name=None, stddev=1.0):
    w,b = norm_weights(fan_out=output_dim,fan_in=input.shape[0],name=name,layer_num=1,bias=True,std=1.0)
    return (T.dot(input, w) + b)

def discriminator(input, h_dim, minibatch_layer=True):
    print(input)
    h0_in = linear(input, h_dim * 2, name = 'd0')
    h0 = T.tanh(h0_in)
    print("Input shape : ",h0_in)

    h1_in = linear(h0, h_dim , name = 'd1')
    h1 = T.tanh(h1_in)
    print("Hidden 1 shape : ",h1_in)

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h_dim)
        print ("h2 Minibatch shape : ",h2.shape)
    # without the minibatch layer, the discriminator )
    else:
        h2 = T.tanh(linear(h_dim, h_dim, name='d2'))
        print("h2 normal batch shape : ", h2.shape)

    h3 = T.nnet.sigmoid(linear(h_dim, 1, name='d3'))
    print("h3 shape :", h3.shape)
    return h3

def get_inputs():
    u = T.imatrix('u')
    # target (where first dimension is time)
    t = T.imatrix('t')
    # initial hidden state of the RNN
    h0 = T.ivector('h0')
    # learning rate
    lr = T.scalar('lr')
    return [h0, u, t, lr]

def sgd(lr, tparams, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.

    #  I HAVE USED GRADS INSIDE FUNCTION HERE
    grads = T.grad(cost, tparams)

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')
    return f_update


def debug_types(loss, params,d_loss_wrt_params, learning_rate=0.01):
    print ("Input type : ", str(type(input)))
    print ("Learning rate type : " + str(type(learning_rate)))
    print ("Derivative Loss wrt parameters : " + str(type(d_loss_wrt_params)), "  Length : ", len(d_loss_wrt_params))
    print ("Parameters : " + str(type(params)), "  Length : ", len(params))
    print(params)
    print ("Loss : " + str(type(loss)))

def debug_variables(loss, input,d_loss_wrt_params, updates):
    print("D loss wrt params ", d_loss_wrt_params)
    print("Input : ", input)
    print("Loss : ", loss)
    print("Updates : ", updates)

def optimizer(loss, params,input,learning_rate=0.01):

    decay = 0.95
    num_decay_steps = 150
    batch = theano.shared(0)
    # uncomment this when fixed !
    # compute gradient of loss with respect to params

    dW = T.grad(loss, params['W'])
    dW_in = T.grad(loss, params['W_in'])
    dW_out = T.grad(loss, params['W_out'])

    #d_loss_wrt_params = T.grad(loss, [params['W'],params['W_in'],params['W_out']])
    # compile the MSGD step into a theano function

    # ValueError: The updates parameter must be an OrderedDict/dict or a list of lists/tuples with 2 elements
    # consider using the sgd function as I'm struggling to resolve why I cannot get gradient descent working
    #  here :/ e.g def sgd(lr, tparams, grads, x, mask, y, cost)

    # Problem here is that the input should contain all the lstm connections:
    # [h0, u, t, lr]  = previous h0 hidden connection, u input connection, time t and learning rate lr
    #print (input)

    gradients = theano.function(inputs=input,outputs=loss, updates=
                        OrderedDict([(params['W'], params['W'] - learning_rate * dW),
                         (params['W_in'], params['W_in'] - learning_rate * dW_in),
                         (params['W_out'], params['W_out'] - learning_rate * dW_out)]),on_unused_input='ignore')

    '''
    train_model = theano.function(inputs=[index, l_r, mom],outputs=cost,updates=updates,
        givens={self.x: train_set_x[index],self.y: train_set_y[index]},mode=mode)
    '''

    return gradients

def minibatch(input, num_kernels=5, kernel_dim=3):

    # this shared variable will have dimensions 4 * 15
    x = theano.shared(input,num_kernels * kernel_dim, dtype = 'float32')
    print(x.shape)
    #  reshaped to 5 *
    activation = np.reshape(x, (-1, num_kernels, kernel_dim))
    print(activation.shape)
    diffs =  (np.expand_dims(activation,3) - np.expand(T.transpose(activation, [1,2,0], 0)))
    abs_diffs = T.sum(T.abs_(diffs),axis=2)
    minibatch_features = T.sum(T.exp(-abs_diffs), 2)
    return T.concatenate([input, minibatch_features],axis=1)

def step(u_t, h_tm1, W, W_in, W_out):
    h_t = T.tanh(T.dot(u_t, W_in) + T.dot(h_tm1, W))
    y_t = T.dot(h_t, W_out)
    return h_t, y_t

class GAN:
    def __init__(self,data,gen,num_steps,batch_size,minibatch,
                 log_every,anim_path,mlp_hidden_size=4):

        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.minibatch = minibatch
        self.log_every = log_every
        self.mlp_hidden_size = mlp_hidden_size
        self.anim_path = anim_path
        self.anim_frames = []
        self.h = OrderedDict()

        if self.minibatch:
            self.learning_rate = 0.001
        else:
            self.learning_rate = 0.05

        self._create_model()

    def _feedforward(self, W, b):
        for i in range(self.mlp_hidden_size):
            if i == 0:
                self.h[i] = T.nnet.softplus(T.dot(self.X, self.w[i]) + self.b[i], name='h' + str(i))
            else:
                self.h[i] = T.nnet.softplus(T.dot(self.h[i], self.w[i]) + self.b[i], name='h' + str(i))
        self.h[len(self.mlp_hidden_size)] = T.nnet.relu(len(self.mlp_hidden_size) - 1, 0)
        return self.h

    def _rnn_weights(self,n,nin,nout):
        disc_weights = OrderedDict()
        disc_weights['W'] = theano.shared(np.random.uniform(size=(n, n), low=-.01, high=.01).astype(theano.config.floatX),borrow=True)
        disc_weights['b'] = theano.shared(np.ones((n, 1)).astype(theano.config.floatX),borrow=True)

        # input to hidden layer weights
        disc_weights['W_in'] = theano.shared(np.random.uniform(size=(nin, n), low=-.01, high=.01).astype(theano.config.floatX),borrow=True)
        disc_weights['b_in'] = theano.shared(np.ones((nin, 1)).astype(theano.config.floatX),borrow=True)

        # hidden to output layer weights
        disc_weights['W_out'] = theano.shared(np.random.uniform(size=(n, nout), low=-.01, high=.01).astype(theano.config.floatX),borrow=True)
        disc_weights['b_out'] = theano.shared(np.ones((nout, 1)).astype(theano.config.floatX),borrow=True)

        return disc_weights

    def _create_model(self):

        '''
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size, self.minibatch)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)
        '''

        self.pre_input = theano.shared(
            value= norm_weight(self.batch_size,1),
            name='pre_input',
            borrow=True
        )

        self.pre_labels = theano.shared(
            value=norm_weight(self.batch_size, 1),
            name='pre_labels',
            borrow=True
        )

        # n = 12, nin = 2*4 = 8, nout = 1
        # seq = np.random.randn(n_seq, n_steps, n_in)

        D_pre_inputs = get_inputs()

        D_pre_weights = self._rnn_weights(self.pre_input.get_value().shape[0], 2*self.mlp_hidden_size, 1)


        # 'pre' indicates that we are pretraining the discriminator in order for this to
        # work properly with the generator, otherwise the generator tricks the discriminaor too easily.

        [d_h_pre, d_y_pre], _ = theano.scan(step,
                                sequences=self.pre_input,
                                outputs_info=[self.pre_labels, None],
                                non_sequences=[D_pre_weights['W'], D_pre_weights['W_in'], D_pre_weights['W_out']])

        #D_pre = discriminator(self.pre_input.get_value(), self.mlp_hidden_size, self.minibatch)

        # 'pre loss' is the mean squared error. This makes sense since we are trying to regenerate the
        # continuous word embeddings themselves. In contrast, if the task was to generate discrete variables
        # we would use negative log likelihood i.e categorical cross entropy

        self.pre_loss = T.mean(T.sqr(d_y_pre - self.pre_labels))

        # In the case of trying to regenerate a discrete sample where we are building a generative model
        # over a discrete input distribution we might consider maximizing the negative log-likelihood instead
        # self.pre_loss = -T.mean(T.log(d_y_pre)+T.log(1-d_y_pre))

        # (loss, params,input,learning_rate=0.01)
        # Problem here is that D_pre_inputs are not passing actual values just placeholders

        # <theano.compile.function_module.Function object at 0x0000000102877F60>
        self.pre_opt = optimizer(self.pre_loss, D_pre_weights, D_pre_inputs, self.learning_rate)
        #OR
        #self.pre_opt = sgd(self.learning_rate, D_pre_weights,D_pre_inputs, self.minibatch, d_y_pre, self.pre_loss)

        self.z = theano.shared(
            value=norm_weight(self.batch_size, 1,).astype(theano.config.floatX),
            name='z',
            borrow=True
        )

        #self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        #self.G = generator(self.z, self.mlp_hidden_size)
        # def generator(input, h_dim):
        # h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
        #h1 = linear(h0, 1, 'g1')

        self.g = T.matrix() #  shape=(self.batch_size, 1) self.z, self.mlp_hidden_size
        gen = Generator(self.z,self.mlp_hidden_size*2)
        self.G = gen.generator()

        # pretraining the discrimintor
        self.x = theano.shared(
            value=norm_weight(self.batch_size, 1,).astype(theano.config.floatX),
            name='x',
            borrow=True
        )

         # n_in, n_hidden, n_out
        print(self.x.get_value().shape)
        d1_discrim = Discriminator(X=self.x,n_in= self.x.get_value().shape[0],
                                   n_hidden = self.mlp_hidden_size,n_out=1, minibatches=self.minibatch)
        _,self.D1 = d1_discrim.rnn_discriminator()
        #self.D1  = discriminator(self.x, self.mlp_hidden_size, self.minibatch)



        # d2_discrim = Discriminator(self.x, self.mlp_hidden_size, self.minibatch)
        # d2_discrim.discriminator()
        d2_discrim = Discriminator(X=self.G,n_in= self.x.get_value().shape[0],
                                   n_hidden = self.mlp_hidden_size,n_out=1, minibatches=self.minibatch)
        _,self.D2 = d2_discrim.rnn_discriminator()

        self.loss_d = disciminator_loss(self.D1,self.D2)
        print("Discriminator loss  : ", self.loss_d)
        self.loss_g = -T.log(self.D2)

        self.d_pre_params = d_y_pre
        # Not sure what the equivalent is to get trainable variables for all scope including D1 and D2
        self.d_params = self.D1
        self.g_params = self.G
        
        #self.opt_d = optimizer(self.loss_d, self.d_params,self.x.get_value(), self.learning_rate)
        #self.opt_g = optimizer(self.loss_g, self.g_params,self.x.get_value(), self.learning_rate)


    def update_discrim(self):
        for step in range(self.num_steps):
            # update discriminator
            x = self.data.sample(self.batch_size)
            z = self.gen.sample(self.batch_size)
            loss_d, _ = [self.loss_d, self.opt_d], {
                self.x: np.reshape(x, (self.batch_size, 1)),
                self.z: np.reshape(z, (self.batch_size, 1))
            }

            # update generator
            z = self.gen.sample(self.batch_size)
            loss_g, _ = [self.loss_g, self.opt_g], {
                self.z: np.reshape(z, (self.batch_size, 1))
            }

            if step % self.log_every == 0:
                print('{}: {}\t{}'.format(step, loss_d, loss_g))

            if self.anim_path:
                self.anim_frames.append(self._samples)

    def train(self):

        num_pretrain_steps = 1000
        for step in range(num_pretrain_steps):
            d = (np.random.random(self.batch_size) - 0.5) * 10.0
            labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
            pretrain_loss, _ = [self.pre_loss, self.pre_opt], {
                self.pre_input: np.reshape(d, (self.batch_size, 1)),
                self.pre_labels: np.reshape(labels, (self.batch_size, 1))
            }
        self.weightsD = self.d_pre_params
        # copy weights from pre-training over to new D network

        for (i,v) in enumerate(self.d_params):
            v.assign(self.weightsD[i])

        self.update_discrim()

    def _samples(self,session,num_points=10000, num_bins=100):
        '''
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        '''

        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # decision boundary
        db = np.zeros((num_points, 1))

        for i in range(num_points // self.batch_size):
            db[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.D1, {
                self.x: np.reshape(
                    xs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)
        return db, pd, pg

    def _plot_distributions(self, session):
        db, pd, pg = self._samples(session)
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db, label='decision boundary')
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()

    def _save_animation(self):
        f, ax = plt.subplots(figsize=(6, 4))
        f.suptitle('1D Generative Adversarial Network', fontsize=15)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1.4)
        line_db, = ax.plot([], [], label='decision boundary')
        line_pd, = ax.plot([], [], label='real data')
        line_pg, = ax.plot([], [], label='generated data')
        frame_number = ax.text(
            0.02,
            0.95,
            '',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
        ax.legend()

        db, pd, _ = self.anim_frames[0]
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))

        def init():
            line_db.set_data([], [])
            line_pd.set_data([], [])
            line_pg.set_data([], [])
            frame_number.set_text('')
            return (line_db, line_pd, line_pg, frame_number)

        def animate(i):
            frame_number.set_text(
                'Frame: {}/{}'.format(i, len(self.anim_frames))
            )
            db, pd, pg = self.anim_frames[i]
            line_db.set_data(db_x, db)
            line_pd.set_data(p_x, pd)
            line_pg.set_data(p_x, pg)
            return (line_db, line_pd, line_pg, frame_number)

        anim = animation.FuncAnimation(
            f,
            animate,
            init_func=init,
            frames=len(self.anim_frames),
            blit=True
        )
        anim.save(self.anim_path, fps=30, extra_args=['-vcodec', 'libx264'])

def main(args):
    model = GAN(DataDistribution(),
                GenerationDistribution(range=8),
                args.num_steps,
                args.batch_size,
                args.minibatch,
                args.log_every,
                args.anim)

    model.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--minibatch', type=bool, default=False,
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim', type=str, default='animation',
                        help='name of the output animation file (default: none)')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
