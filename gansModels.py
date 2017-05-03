from keras.layers import Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge,\
     LSTM, Bidirectional, Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers.core import Flatten
from keras.optimizers import SGD

def basic_cnn():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(1, 50, 50)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def basicgen_cnn():
    model = Sequential()
    model.add(Dense(input_dim=50, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(50 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((50, 25, 25), input_shape=(50 * 7 * 7,)))
    return model


def my_cnn(params, emb_layer=True):
    nb_feature_maps = 100;
    filter_sizes = (3, 4);
    model = Sequential()
    if emb_layer:
        model.add(Embedding(input_dim=params['weights'].shape[0], output_dim=params['weights'].shape[1],
                            weights=[params['weights']]))
        model.add(Convolution1D(nb_filter=nb_feature_maps, filter_length=1,
                                border_mode='valid', activation='tanh'))
    else:
        model.add(Convolution1D(nb_filter=nb_feature_maps, filter_length=1,
                                border_mode='valid', activation='tanh', input_shape=params['weights'][0].shape))
    model.add(MaxPooling1D(pool_length=50))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(100))
    model.add(Activation('tanh'))
    model.add(Dense(len(params['num_class'])))
    model.add(Activation('sigmoid'))
    return model


def mycnn_gen(params, emb_layer=False):
    nb_feature_maps = 1000;
    n_gram = 2;
    model = Sequential()
    if emb_layer:
        model.add(Embedding(input_dim=params['weights'].shape[0], output_dim=params['weights'].shape[1],
                            weights=[params['weights']]))
        model.add(Convolution1D(nb_filter=nb_feature_maps, filter_length=1,
                                border_mode='valid', activation='tanh'))
    else:
        model.add(Convolution1D(nb_filter=nb_feature_maps, filter_length=1,
                                border_mode='valid', activation='tanh', input_shape=params['weights'][0].shape))
    model.add(MaxPooling1D(pool_length=50))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    return model


def kim_cnn(params, type='discriminator'):
    sequence_length = params['data_train'][0].shape[0]
    embedding_dim = params['data_train'][0].shape[1]
    filter_sizes = (3, 4)
    num_filters = 150
    dropout_prob = (0.25, 0.5)
    # be careful, this might have to be changed to 150 when concatenated in this next layer (150 x 2)
    hidden_dims = 300
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='tanh',
                             subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)
    # main sequential model
    model = Sequential()
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    if type is 'discriminator':
        model.add(Activation('tanh'))
    else:
        model.add(Activation('tanh'))
        model.add(Dense(1))
        # model.add(Dense(len(params['num_class'])))
        model.add(Activation('sigmoid'))
    return model


def kimgen_cnn(params):
    sequence_length = params['data_train'][0].shape[0]
    embedding_dim = params['data_train'][0].shape[1]
    filter_sizes = (3, 4)
    num_filters = 150
    dropout_prob = (0.25, 0.5)
    hidden_dims = 150
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='tanh',
                             subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)
    # main sequential model
    model = Sequential()
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    # model.add(Dense(len(params['num_class'])))
    model.add(Activation('sigmoid'))
    return model

def lstm_generator_model(params, bi=False):
    model = Sequential()
    model.add(Embedding(input_dim=params['weights'].shape[0], output_dim=params['weights'].shape[1],
                        weights=[params['weights']]))
    if bi:
        model.add(Bidirectional(LSTM(output_dim=len(params['num_class']),
                                     activation='tanh', input_shape=params['data_train'][0].shape)))
    else:
        model.add(LSTM(output_dim=len(params['num_class']),
                       activation='tanh', input_shape=params['data_train'][0].shape))
    model.add(Dense(len(params['num_class'])))
    model.add(Activation('tanh'))
    return model

def lstm_discriminator_model(params,bi=False):
    model = Sequential()
    model.add(Embedding(input_dim=params['weights'].shape[0],
                        output_dim=params['weights'].shape[1],
                        weights=[params['weights']]))
    if bi:
        model.add(
            Bidirectional(LSTM(output_dim=len(params['num_class']),
                               activation='tanh',
                            input_shape=params['data_train'][0].shape)))
    else:
        model.add(
            LSTM(output_dim=len(params['num_class']),
                 activation='tanh',
                 input_shape=params['data_train'][0].shape))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
