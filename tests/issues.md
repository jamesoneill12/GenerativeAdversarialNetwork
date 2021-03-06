# Issues to be resolved

### 1. Assigning discriminator parameters with pretrained values

In def train() self.weightsD parameters need to be the same of the weights assigned to the main discriminator parameters self.d_params.
Otherwise there is a mismatch and it will not work. 

Therefore, the LSTM architecture used for the pretrained discriminator should also be applied for the main one. At the moment they
are both different as the main discriminator is a simple feedforward network.

## Pretrained discriminator weights (8) 

X, n_in, n_hidden, n_out are main dimensions for designing weights. 
* X = (12,1) batch size default arg is (12,1)
* n_in = (12,1) batch size default arg is 12
* n_hidden = 4
* n_out = 1

\# recurrent weights as a shared variable
<pre><code>
W_init = (4,4)
self.W = theano.shared(value=W_init, name='W')

\# input to hidden layer weights
W_in_init = (12,4)
self.W_in = theano.shared(value=W_in_init, name='W_in')

\# hidden to output layer weights
W_out_init = (4,1)
self.W_out = theano.shared(value=W_out_init, name='W_out')

h0_init = (4,) of zeros
self.h0 = theano.shared(value=h0_init, name='h0')

bh_init = (4,) of zeros
self.bh = theano.shared(value=bh_init, name='bh')

by_init = (1,) of zeros
self.by = theano.shared(value=by_init, name='by')
</code></pre>

**self.params = {'W' : (4,4), 'W_in' : (12,4), 'W_out' : (4,1), 'h0' : (4,), 'bh' : (4,), 'by' : (1,)}**

## Function output shapes
* (12,8)
* (12,8)
* (12,13)
* (12,1)
## Weights shapes in each layer
* w (8,8) ,b (8,)
* w (8,15) ,b (15,)
* w (13,) ,b (1,)
* w (1,4) ,b (4,)

## Discriminator weights (8)

* n = batch_size = 12
* n_in = 2 * mlp hidden size = 8
* nout = 1

<pre><code>
\# def _rnn_weights(self,n_in,n,nout):
    disc_weights = OrderedDict()
    disc_weights['W'] = theano.shared(np.random.uniform(size=(n, n), low=-.01, high=.01).astype(theano.config.floatX),borrow=True)
    disc_weights['b'] = theano.shared(np.ones((n, 1)).astype(theano.config.floatX),borrow=True)

    \# input to hidden layer weights
    disc_weights['W_in'] = theano.shared(np.random.uniform(size=(nin, n), low=-.01, high=.01).astype(theano.config.floatX),borrow=True)
    disc_weights['b_in'] = theano.shared(np.ones((nin, 1)).astype(theano.config.floatX),borrow=True)

    \# hidden to output layer weights
    disc_weights['W_out'] = theano.shared(np.random.uniform(size=(n, nout), low=-.01, high=.01).astype(theano.config.floatX),borrow=True)
    disc_weights['b_out'] = theano.shared(np.ones((nout, 1)).astype(theano.config.floatX),borrow=True)
</code></pre>

**self.params = {'W' : (12,12), 'b' : (12,1), 'W_in' : (8,12), 'b_in' : (12,1), 'W_out' : (12,1), 'b_out' : (1,1)}**

Function output shapes
* (12,8)
* (12,8)
* (12,13)
* (12,1)
Weights shapes in each layer
* w (8,8) ,b (8,)
* w (8,12) ,b (12,)
* w (13,) ,b (1,)
* w (12,1) ,b (1,)

## Generator weights (6)

Function output shapes
*(12,8)
* (12,8)
* (12,13)
* (12,1)
Weights shapes in each layer
* w (8,8) ,b (8,)
* w (8,15) ,b (15,)
* w (13,) ,b (1,)
* w (1,8) ,b (8,)**

### 2. Animation of the generated distribution

At the moment I need to download the software that will save the images in bitmap form which then can be saved as a video. 
This is useful for visualization purposes.
