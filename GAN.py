import argparse
import math
from helpers import *
from gansModels import *

# Output of the discriminator should be 7L, not 150L.
# Some mistake here that needs to be resolved !

def cnn_discriminator_model(params,type='kim',kimgen=False,emb_layer=False):
    if type is 'my':
        model = my_cnn(params)
    elif type is 'basic':
        model = basic_cnn()
    else:
        model = kim_cnn(params)
    return model

def cnn_generator_model(params, type='kim', emb_layer=False):
    if type is 'my':
        model = mycnn_gen(params)
    elif type is 'basic':
        model = basicgen_cnn()
    else:
        model = kimgen_cnn(params)
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image

def train(BATCH_SIZE,datapath=None):
    if datapath==None:
        datapath = "C:/Users/1/James/grctc/GRCTC_Project/Classification/" \
                   "Extract_Documents/resources/en/eurovoc_subset_all.txt"
    params = data_split(datapath,field='subject',multilabel=True)
    (X_train, y_train) = (params['data_train'],params['labels_train'])
    (X_test, y_test) = (params['data_test'],params['labels_test'])
    print X_train[0].shape
    print y_train[0].shape
    #X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    params['xtrain_reshape'] = X_train.shape
    print (X_train.shape)
    print (y_train.shape)
    discriminator = cnn_discriminator_model(params)#params,bi=True)
    generator = cnn_generator_model(params)#params,bi=True)
    print ("Generator input shape : ", generator.input_shape)
    print ("Discriminator input shape : ", discriminator.input_shape)
    print ("Generator output shape : ", generator.output_shape)
    print ("Discriminator output shape : ", discriminator.output_shape)
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    print ("Generator/discriminator input shape : ",discriminator_on_generator.input_shape)
    print ("Generator/discriminator output shape : ",discriminator_on_generator.output_shape)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 50))
    for epoch in range(10):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, X_train.shape[1])
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)
            print image_batch.shape
            print generated_images.shape
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 50)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    generator = cnn_generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator =cnn_discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE*20, 50))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 50)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 50))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 50)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)

args = get_args()
train(BATCH_SIZE=args.batch_size)
