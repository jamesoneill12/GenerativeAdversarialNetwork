# Description of the data shapes pass for pretrained discriminator, discriminators and generator

## Pretrained Discriminator
Input shape :  (12, 8)
Hidden 1 shape :  (12, 8)
Minibatch h2 shape :  (12, 13)
output shape  (12, 1)

### Discriminator 1
Input shape :  (12, 8)
Hidden 1 shape :  (12, 8)
Minibatch h2 shape :  (12, 13)
output shape  (12, 1)

### Discriminator 2
Input shape :  (12, 8)
Hidden 1 shape :  (12, 8)
Minibatch h2 shape :  (12, 13)
output shape  (12, 1)

## Generator  
input shape (12, 4)
Hidden 1 shape :  (12, 1)

## Current problem

####### Arguments for the optimizer #######
('Pre loss', mean)
('D_pre_weights', OrderedDict([('W', <CudaNdarrayType(float32, matrix)>), ('b', <CudaNdarrayType(float32, matrix)>), ('W_in', <CudaNdarrayType(float32, matrix)>), ('b_in', <CudaNdarrayType(float32, matrix)>), ('W_out', <CudaNdarrayType(float32, matrix)>), ('b_out', <CudaNdarrayType(float32, matrix)>)]))
('D_pre_inputs', [h0, u, t, lr])
('Fan in : ', 8)
('Fan out : ', 12)
####### Arguments for the updater #######
('D1 loss', mean)
('D1 Discriminator parameters', {'h0': h0, 'bh': bh, 'W_out': W_out, 'W': W, 'by': by, 'W_in': W_in})
('D1 inputs', [h0, u, t, lr])
<class 'theano.sandbox.cuda.var.CudaNdarraySharedVariable'>
('self.x is of type ', <class 'theano.sandbox.cuda.var.CudaNdarraySharedVariable'>)
x
('self.G is of type ', <class 'theano.tensor.var.TensorVariable'>)
Elemwise{mul,no_inplace}.0
<class 'theano.sandbox.cuda.var.CudaNdarraySharedVariable'>
####### Discriminator Outputs #######
('Discriminator 1 output : ', <theano.compile.function_module.Function object at 0x000000007FAFBEB8>)
('Discriminator 2 output : ', <theano.compile.function_module.Function object at 0x00000000F61AFEB8>)
