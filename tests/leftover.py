'''
hidden_size = 10
x = T.vector('X', dtype='float32')
z = T.vector('G', dtype='float32')
x= np.random.normal(size=(3,2))
z = np.random.normal(size=(3,2))

G = Generator(z,hidden_size)
gen = G.generator()
D1 = Discriminator(z,hidden_size)
D1s = D1.discriminator()
D2 = Discriminator(G,hidden_size)
D2s = D2.discriminator()

d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G/')]
print([D1s.w,D2s.w])
print ([G.w])

opt_d = optimizer(disciminator_loss(D1,D2), d_params)
opt_g = optimizer(generator_loss(D2), g_params)

#results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym), sequences=X)
'''
