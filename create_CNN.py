from dataset import DataLoader
from keras.models import Model, load_model
import os
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,  BatchNormalization,Dropout, Flatten, Dense
from keras import callbacks
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt

class ModelTrainer:

    def __init__(self):
        self.testset_creator= DataLoader()

    def load_trainig_data(self,img_height, img_width, directory='Images/chihuahua-muffin/train', split_factor=0.4):
        images, labels = self.testset_creator.load_data(directory, img_width,img_height)
        images, labels = self.testset_creator.normalizeData(images, labels)
        images, labels = self.testset_creator.shuffel_data(images, labels)
        print('X.shape: ',images.shape)
        print('Y.shape: ', labels.shape)

        train_X, Test_img, train_Y, Test_label = self.testset_creator.cross_validate(images, labels, split_factor)
        val_X = Test_img[0:int(len(Test_img)*0.5)]
        val_Y = Test_label[0:int(len(Test_label)*0.5)]
        test_X= Test_img[int(len(Test_img)*0.5):]
        test_Y = Test_label[int(len(Test_label) * 0.5):]
        return train_X, val_X, train_Y, val_Y, test_X, test_Y

    def create_model(self, img_size_X, img_size_Y):
        model = Sequential()
        model.add(Convolution2D(16, (5, 5), activation='relu',kernel_initializer='glorot_uniform', padding='same', input_shape=(img_size_X, img_size_Y, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
        # model.add(BatchNormalization())
        # model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
        # model.add(BatchNormalization())
        model.add(Convolution2D(16, (3, 3),kernel_initializer='glorot_uniform', activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(16, (3, 3),kernel_initializer='glorot_uniform', activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512,kernel_initializer='glorot_uniform', activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(128,kernel_initializer='glorot_uniform', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))

        return model

    def compile_model(self, model, optimizer, loss_function='binary_crossentropy'):
        model.compile(optimizer=optimizer, loss=loss_function,  metrics=['accuracy'])
        model.summary()
        return model

    def __create_callbacks(self):
        filepath = 'test-{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                               save_weights_only=False, mode='auto', period=1)

        early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=200,
                                             verbose=0, mode='auto')

        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                patience=3, min_lr=0.00001)

        tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32,
                                            write_graph=True, write_grads=False, write_images=False,
                                            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


        callbacks_list = [reduce_lr, tensorboard, checkpoint]
        return callbacks_list

    def train_model(self, model, train_data, train_label, validation_data, validation_label, batch_size=4, epoch=100,):
        model.fit(train_data, train_label,
                  batch_size=batch_size,
                  nb_epoch=epoch,
                  validation_data=(validation_data, validation_label),
                  verbose=2, callbacks=self.__create_callbacks())

    def plot_model_history(self, model_history):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # summarize history for accuracy
        axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
        axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
        axs[0].legend(['train', 'val'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
        axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
        axs[1].legend(['train', 'val'], loc='best')
        plt.show()

    def plot_images(self):

        images,labels = self.testset_creator.load_data('Images/chihuahua-muffin/test', 170,170)

        class_names = ['dogs', 'cats']
        fig = plt.figure(figsize=(4, 3))
        for index,img in enumerate(images):
            ax = fig.add_subplot(4, 4, 1 + index, xticks=[], yticks=[])
            im = img
            ax.set_xlabel('zeile1 \n zeile2', fontsize=12)
            plt.imshow(im)

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    model_trainer= ModelTrainer()
    img_height, img_width=150, 150
    #model= model_trainer.create_model( img_height, img_width)
    #model= model_trainer.compile_model(model, SGD(lr=0.001, momentum=0.9), 'categorical_crossentropy')

    #train_X, val_X, train_Y, val_Y, test_X, test_Y =  model_trainer.load_trainig_data( img_height, img_width, directory='Images/chihuahua-muffin/test')
    #model_trainer.train_model(model,train_X, train_Y, val_X, val_Y, 32, 500)
    #model.save('my_model.hdf5')
    model_trainer.plot_images()

    # train_X, val_X, train_Y, val_Y, test_X, test_Y = model_trainer.load_trainig_data( img_height, img_width, 'Images/chihuahua-muffin/test', 0 )
    # test_X= train_X
    # test_Y =train_Y
    # model=load_model('bassd-schoh.hdf5',compile=True)
    # predicted_classes=model.predict_classes(test_X)
    # print(predicted_classes)

    # from confusion_Matrix import Confusionmatrix
    # cnf=Confusionmatrix(['dogs','muffins'])
    # cnf_matriy=cnf.compute_confusion_matrix(test_Y,predicted_classes)
    # cnf.plot_confusion_matrix(cnf_matriy,True, title='150px')
    # train_features= train_X
    # train_labels=train_Y
    # import matplotlib.pyplot as plt
    # #(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
    # num_train, img_channels, img_rows, img_cols = train_features.shape
    # num_test, _, _, _ = train_features.shape
    # num_classes = len(np.unique(train_labels))

    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #                'dog', 'frog', 'horse', 'ship', 'truck']
    # import cv2
    # class_names = ['dogs', 'cats']
    # fig = plt.figure(figsize=(6, 4))
    # for i in range(num_classes):
    #     ax = fig.add_subplot(1, 2, 1 + i, xticks=[], yticks=[])
    #     idx = np.where(train_labels[:] == i)[0]
    #     features_idx = train_features[idx, ::]
    #     img_num = np.random.randint(features_idx.shape[0])
    #
    #     im = features_idx[img_num, ::]
    #     ax.set_title(class_names[i])
    #     ax.set_xlabel('zeile1 \n zeile2', fontsize=12)
    #     plt.imshow(im)
    # plt.show()

    #train_features = train_features.astype('float32') / 255
    #test_features = test_features.astype('float32') / 255
    # convert class labels to binary class labels
    #train_labels = np_utils.to_categorical(train_labels, num_classes)
    #test_labels = np_utils.to_categorical(test_labels, num_classes)

    # from sklearn.model_selection import train_test_split
    # X_train, val_X, y_train, val_Y = train_test_split(train_features, train_labels, test_size=0.2, random_state=0)

    # train_Y= np_utils.to_categorical(train_Y, num_classes)
    # val_Y = np_utils.to_categorical(val_Y, num_classes)
    # history= model_trainer.train_model(model,train_X, train_Y, val_X, val_Y, 32, 500)


