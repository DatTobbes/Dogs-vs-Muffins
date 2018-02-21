from dataset import DataLoader
from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import os
from keras import callbacks
import numpy as np
from keras.optimizers import SGD


class ModelTrainer:

    def __init__(self):
        self.testset_creator= DataLoader()

    def load_trainig_data(self,img_height, img_width, directory='Images/chihuahua-muffin/train'):
        images, labels = self.testset_creator.load_data(directory, img_width,img_height)
        images, labels = self.testset_creator.normalizeData(images, labels)
        print('X.shape: ',images.shape)
        print('Y.shape: ', labels.shape)

        train_X, val_X, train_Y, val_Y = self.testset_creator.cross_validate(images, labels,0.2)
        return train_X, val_X, train_Y, val_Y

    def create_model(self, img_size_X, img_size_Y):
        img_in = Input(shape=(img_size_X, img_size_Y, 3), name='img_in')

        x = Convolution2D(32, (5, 5), padding='same')(img_in)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Convolution2D(32, (5, 5), padding='same')(img_in)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Convolution2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Convolution2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        merged = Flatten()(x)

        x = Dense(64, activation='relu')(merged)
        x = Dropout(.2)(x)
        class_out = Dense(1, name='Class_Out', activation='sigmoid')(x)

        model = Model(input=[img_in], output=[class_out])
        return model

    def compile_model(self, model, optimizer, loss_function='binary_crossentropy'):
        model.compile(optimizer=optimizer, loss=loss_function,  metrics=['accuracy'])
        model.summary()
        return model

    def __create_callbacks(self):
        filepath = 'RopePrediction-{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                               save_weights_only=False, mode='auto', period=1)

        early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100,
                                             verbose=0, mode='auto')

        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                patience=3, min_lr=0.00001)

        tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32,
                                            write_graph=True, write_grads=False, write_images=False,
                                            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

        csvLogger = callbacks.CSVLogger('C:/tmp\LearningLogs/log.csv', separator=',', append=False)

        callbacks_list = [ reduce_lr, tensorboard]
        return callbacks_list

    def train_model(self, model, train_data, train_label, validation_data, validation_label, batch_size=4, epoch=100,):
        model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=epoch, validation_data=(validation_data, validation_label), verbose=2, callbacks=self.__create_callbacks())


if __name__ == '__main__':
    model_trainer= ModelTrainer()
    model= model_trainer.create_model(171,172)
    model= model_trainer.compile_model(model,SGD(lr=0.001, momentum=0.9),'binary_crossentropy')

    train_X, val_X, train_Y, val_Y =  model_trainer.load_trainig_data(171,172)

    model_trainer.train_model(model,train_X, train_Y, val_X, val_Y, 16, 1000 )