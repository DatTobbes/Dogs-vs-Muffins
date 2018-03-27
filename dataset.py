import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from image_augmentation import Img_Augmentation

class DataLoader:

    def __init__(self):
        self.images= []
        self.labels=[]
        self.augmenttator= Img_Augmentation()

    def save_as_npz(self, images, labels, filename):
        np.savez_compressed( filename,
                            images=images,
                            labels=labels)

    def load_data(self, data_dir, img_Size_X, img_size_Y):
        directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
        images = []
        labels = []
        category = 0
        for d in directories:
            file_names = []
            label_dir = os.path.join(data_dir, d)
            for path, subdirs, files in os.walk(label_dir):
                for name in files:
                    if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg"):
                        os.path.join(path, name)
                        file_names.append(os.path.join(path, name))

            for f in file_names:
                img = cv2.imread(f)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imresize = cv2.resize(img, (img_Size_X, img_size_Y))
                images.append(imresize)
                labels.append(category)

            category += 1
        print(len(labels))
        return images, labels

    def argument_images(self,images, labels):

        for index in range(len(images)):
            print(len(images))
            img = images[index]
            flipt_img = np.expand_dims(self.augmenttator.rnd_flip(img), axis=0)
            #gray_scaled_img =  cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #img_gray =self.change_color(img)

            images=np.vstack((images, flipt_img))
            labels = np.append(labels, np.asarray(labels[index]))


        return images, labels

    def change_color(self, img):
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        color_converted = [b, r, g]
        np.asarray(color_converted)
        color_converted = np.reshape(color_converted, (171, 171, 3))

    def printImageAndLabelArray(self):
        print(images, labels)

    def cross_validate(self,ImgArr, LabelArray, testSize=0.2):
        X_train, X_test, y_train, y_test = train_test_split(ImgArr , LabelArray, test_size=testSize, random_state=0)
        return X_train, X_test, y_train, y_test

    def normalizeData(self, ImagArray,LabelArray):
        imageArray= np.array(ImagArray).astype('float32')
        imageArray= imageArray / 255
        labelArray= np.array(LabelArray)
        return imageArray, labelArray

    def shuffel_data(self, img_array, label_array):
        from sklearn.utils import shuffle
        return shuffle(img_array, label_array, random_state=9)


if __name__ == "__main__":
    data_dir = '.\\Images\\chihuahua-muffin\\train'
    testset= DataLoader()
    images, labels = testset.load_data(data_dir, 171,171)
    testset.save_as_npz(images, labels, 'dogs-vs-muffins.npz')
    import matplotlib.pyplot as plt

    dataset = np.load('dogs-vs-muffins.npz')
    labels=dataset['labels']
    images=dataset['images']
    images, labels=testset.argument_images(images, labels)
    images, labels = testset.normalizeData(images, labels)
    images_to_show = np.concatenate((images[labels == 0][:8], images[labels == 1][:8]), axis=0)
    print(len(images_to_show))

    fig = plt.figure(figsize=(10, 10))
    for i in range(16):
         ax = fig.add_subplot(8, 8, 1 + i, xticks=[], yticks=[])
         im = images_to_show[i]
         plt.imshow(im)

    plt.tight_layout()
    plt.show()

