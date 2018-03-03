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
                #self.create_rnd_img(imresize, category, images, labels)

            category += 1
        print(len(labels))
        return images, labels

    def argument_images(self,images, labels):

        for index in range(len(images)):
            print(len(images))
            img = images[index]
            flipt_img = self.augmenttator.rnd_flip(img)
            gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imshow("", gray_scaled_img)
            cv2.waitKey(0)

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

    def create_rnd_img(self, img, label, images,labels):
        new_img=img

        flipt_img=self.augmenttator.rnd_flip(new_img)
        images.append(flipt_img)
        labels.append(label)
        #cv2.imshow('flip', flipt_img)

        # rot_img=self.augmenttator.rnd_rotation(new_img,[0,360])
        # images.append(rot_img)
        # labels.append(label)
        #cv2.imshow('rot', rot_img)

        # flipt_img = self.augmenttator.rnd_flip(new_img)
        # rot_img = self.augmenttator.rnd_rotation(flipt_img, [0,360])
        # images.append(rot_img)
        # labels.append(label)
        #cv2.imshow('image', rot_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


if __name__ == "__main__":
    data_dir = '.\\Images\\chihuahua-muffin\\train'
    #testset= DataLoader()
    #images, labels = testset.load_data(data_dir, 171,172)
    #images, labels = testset.normalizeData(images, labels)
    #testset.save_as_npz(images, labels, 'dog-vs-muffins.npz')
    import matplotlib.pyplot as plt

    #images, labels = testset.argument_images(images , labels)
    testset= DataLoader()
    dataset = np.load('test_data.npz')
    labels, images = [dataset[f] for
                      f in dataset.files]
    images_to_show = np.concatenate((images[labels == 0][:8], images[labels == 1][:8]), axis=0)
    print(len(images_to_show))

    fig = plt.figure(figsize=(10, 10))
    for i in range(16):
        ax = fig.add_subplot(4, 4, 1 + i, xticks=[], yticks=[])
        im = images_to_show[i]
        plt.imshow(im)

    plt.tight_layout()
    plt.show()

