# -*- coding:utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
from tensorflow.keras.models import Sequential

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.preprocessing.image import array_to_img
import cv2
from keras.callbacks import CSVLogger
from data import *
from keras.optimizers import SGD
from keras import backend as K
smooth = 1.
epsilon = 1e-5
smooth = 1

def dice(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice(y_true, y_pred)
    return loss

class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test



    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print(conv1)
        print "conv1 shape:", conv1.shape
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print "conv1 shape:", conv1.shape
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print "pool1 shape:", pool1.shape

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print "conv2 shape:", conv2.shape
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print "conv2 shape:", conv2.shape
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print "pool2 shape:", pool2.shape

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print "conv3 shape:", conv3.shape
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print "conv3 shape:", conv3.shape
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print "pool3 shape:", pool3.shape

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2DTranspose(512, (2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)
        merge6 = concatenate([drop4,up6],axis=3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2DTranspose(256,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        merge7 =concatenate([conv3,up7],axis=3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2DTranspose(128,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        merge8 =concatenate([conv2,up8],axis=3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2DTranspose(64,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        merge9 =concatenate([conv1,up9],axis=3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        print(conv10)
        model = Model(inputs=inputs, outputs=conv10)
        opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss=dice, metrics=[dice_loss])
        #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model



    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
        #model_checkpoint =tf.keras.callbacks.ModelCheckpoint('./exp9/unet{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=False)
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        e =EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        print('Fitting model...')
        csv_logger = CSVLogger('exp1.csv', append=True, separator=';')

        model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=1, verbose=1,
                  validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, csv_logger])


        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('./results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('./results/imgs_mask_test.npy')
        piclist = []
        for line in open("./results/pic.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = imgs[i]
            img = array_to_img(img)
            img.save(path)
            cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            cv_pic = cv2.resize(cv_pic,(780,580),interpolation=cv2.INTER_CUBIC)
            binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(path, cv_save)


if __name__ == '__main__':
    myunet = myUnet()
    model = myunet.get_unet()
    # model.summary()
    # plot_model(model, to_file='model.png')
    myunet.train()
    myunet.save_img()
