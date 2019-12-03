## Assignment-3

**Reported Base Network** :390/390 [==============================] - 20s 51ms/step - loss: 0.3205 - acc: 0.8938 - val_loss: 0.5788 - val_acc: 0.8294
Model took 1012.00 seconds to train
Accuracy on test data is: **82.94**

**New Network**
Accuracy: **86.04**
Total params: 98,797
Trainable params: 97,049
Non-trainable params: 1,748
Epoch:50
Batch Size:512

# Network

mymodel=Sequential()
mymodel.add(SeparableConv2D(32,(3,3),strides=1,padding='same',activation='relu',input_shape=(32, 32, 3))) #32,32,32,3
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.05))

mymodel.add(SeparableConv2D(64,(3,3),padding='same',strides=1,activation='relu'))#32,32,64,5
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.05))

mymodel.add(SeparableConv2D(128,(3,3),padding='same',strides=1,activation='relu'))#32,32,128,7
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.05))

mymodel.add(SeparableConv2D(32,(1,1),padding='same',strides=2,activation='relu'))#16,16,32,7
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.05))

mymodel.add(SeparableConv2D(64,(3,3),padding='same',strides=1,activation='relu'))#16,16,64,11
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(128,(3,3),padding='same',strides=1,activation='relu'))#16,16,128,15
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(32,(1,1),strides=2,padding='same',activation='relu'))#8,8,32,15
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(64,(3,3),padding='same',strides=1,activation='relu'))#8,8,64,23
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(128,(3,3),padding='same',strides=1,activation='relu'))#8,8,128,31
mymodel.add(BatchNormalization())
mymodel.add(Dropout(0.1))

mymodel.add(SeparableConv2D(10,(1,1),strides=2,padding='same',activation='relu'))#4,4,10,39
mymodel.add(BatchNormalization())


mymodel.add(GlobalAveragePooling2D())
mymodel.add(Activation('softmax'))

mymodel.summary()


clr = CyclicLR(
	base_lr=0.0005,
	max_lr=0.09,
	step_size= 811)

mymodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(shear_range=0.1,zoom_range=0.1,horizontal_flip=True)


# Logs

Epoch 1/50
 2/97 [..............................] - ETA: 5:22 - loss: 2.3196 - acc: 0.1064
/usr/local/lib/python3.6/dist-packages/keras/callbacks.py:95: RuntimeWarning: Method (on_train_batch_end) is slow compared to the batch update (0.691678). Check your callbacks.
  % (hook_name, delta_t_median), RuntimeWarning)
97/97 [==============================] - 43s 446ms/step - loss: 1.7865 - acc: 0.3549 - val_loss: 9.7307 - val_acc: 0.2244

Epoch 2/50
97/97 [==============================] - 38s 394ms/step - loss: 1.2999 - acc: 0.5415 - val_loss: 12.1136 - val_acc: 0.2182

Epoch 3/50
97/97 [==============================] - 38s 394ms/step - loss: 1.1197 - acc: 0.6025 - val_loss: 9.4507 - val_acc: 0.3375

Epoch 4/50
97/97 [==============================] - 38s 395ms/step - loss: 1.0264 - acc: 0.6353 - val_loss: 9.7471 - val_acc: 0.2813

Epoch 5/50
97/97 [==============================] - 38s 395ms/step - loss: 0.9669 - acc: 0.6555 - val_loss: 10.3100 - val_acc: 0.2879

Epoch 6/50
97/97 [==============================] - 38s 394ms/step - loss: 0.9091 - acc: 0.6769 - val_loss: 6.4618 - val_acc: 0.4037

Epoch 7/50
97/97 [==============================] - 38s 394ms/step - loss: 0.8604 - acc: 0.6972 - val_loss: 11.0343 - val_acc: 0.2235

Epoch 8/50
97/97 [==============================] - 38s 393ms/step - loss: 0.8236 - acc: 0.7081 - val_loss: 6.8308 - val_acc: 0.4252

Epoch 9/50
97/97 [==============================] - 38s 393ms/step - loss: 0.7833 - acc: 0.7257 - val_loss: 7.5552 - val_acc: 0.3220

Epoch 10/50
97/97 [==============================] - 38s 393ms/step - loss: 0.7367 - acc: 0.7406 - val_loss: 2.6254 - val_acc: 0.5361

Epoch 11/50
97/97 [==============================] - 38s 393ms/step - loss: 0.6927 - acc: 0.7570 - val_loss: 1.2703 - val_acc: 0.6896

Epoch 12/50
97/97 [==============================] - 38s 392ms/step - loss: 0.6544 - acc: 0.7710 - val_loss: 1.2892 - val_acc: 0.6825

Epoch 13/50
97/97 [==============================] - 38s 391ms/step - loss: 0.6263 - acc: 0.7808 - val_loss: 1.0727 - val_acc: 0.6935

Epoch 14/50
97/97 [==============================] - 38s 392ms/step - loss: 0.5985 - acc: 0.7921 - val_loss: 0.7152 - val_acc: 0.7705

Epoch 15/50
97/97 [==============================] - 38s 392ms/step - loss: 0.5613 - acc: 0.8038 - val_loss: 0.6036 - val_acc: 0.8004

Epoch 16/50
97/97 [==============================] - 38s 392ms/step - loss: 0.5379 - acc: 0.8124 - val_loss: 0.5530 - val_acc: 0.8109

Epoch 17/50
97/97 [==============================] - 38s 390ms/step - loss: 0.5108 - acc: 0.8208 - val_loss: 0.5260 - val_acc: 0.8230

Epoch 18/50
97/97 [==============================] - 38s 391ms/step - loss: 0.5148 - acc: 0.8202 - val_loss: 0.5548 - val_acc: 0.8107

Epoch 19/50
97/97 [==============================] - 38s 392ms/step - loss: 0.5291 - acc: 0.8163 - val_loss: 0.6148 - val_acc: 0.7967

Epoch 20/50
97/97 [==============================] - 38s 388ms/step - loss: 0.5674 - acc: 0.8012 - val_loss: 1.1187 - val_acc: 0.6936

Epoch 21/50
97/97 [==============================] - 38s 387ms/step - loss: 0.5810 - acc: 0.7974 - val_loss: 1.2121 - val_acc: 0.6720

Epoch 22/50
97/97 [==============================] - 38s 388ms/step - loss: 0.6125 - acc: 0.7855 - val_loss: 1.8252 - val_acc: 0.6089

Epoch 23/50
97/97 [==============================] - 38s 388ms/step - loss: 0.6325 - acc: 0.7786 - val_loss: 3.9951 - val_acc: 0.4488

Epoch 24/50
97/97 [==============================] - 37s 385ms/step - loss: 0.6342 - acc: 0.7796 - val_loss: 2.9924 - val_acc: 0.5498

Epoch 25/50
97/97 [==============================] - 38s 392ms/step - loss: 0.6310 - acc: 0.7798 - val_loss: 2.7945 - val_acc: 0.5191

Epoch 26/50
97/97 [==============================] - 38s 388ms/step - loss: 0.6210 - acc: 0.7817 - val_loss: 1.5268 - val_acc: 0.6617

Epoch 27/50
97/97 [==============================] - 38s 393ms/step - loss: 0.5906 - acc: 0.7935 - val_loss: 1.5747 - val_acc: 0.6575

Epoch 28/50
97/97 [==============================] - 38s 389ms/step - loss: 0.5626 - acc: 0.8026 - val_loss: 1.0208 - val_acc: 0.7270

Epoch 29/50
97/97 [==============================] - 38s 387ms/step - loss: 0.5371 - acc: 0.8118 - val_loss: 0.6405 - val_acc: 0.8046

Epoch 30/50
97/97 [==============================] - 38s 387ms/step - loss: 0.5070 - acc: 0.8217 - val_loss: 1.1461 - val_acc: 0.6968

Epoch 31/50
97/97 [==============================] - 38s 388ms/step - loss: 0.4923 - acc: 0.8267 - val_loss: 0.6515 - val_acc: 0.7921

Epoch 32/50
97/97 [==============================] - 37s 386ms/step - loss: 0.4664 - acc: 0.8370 - val_loss: 0.5257 - val_acc: 0.8275

Epoch 33/50
97/97 [==============================] - 37s 385ms/step - loss: 0.4384 - acc: 0.8457 - val_loss: 0.5010 - val_acc: 0.8359

Epoch 34/50
97/97 [==============================] - 37s 385ms/step - loss: 0.4264 - acc: 0.8514 - val_loss: 0.4783 - val_acc: 0.8411

Epoch 35/50
97/97 [==============================] - 39s 399ms/step - loss: 0.4363 - acc: 0.8469 - val_loss: 0.5773 - val_acc: 0.8051

Epoch 36/50
97/97 [==============================] - 37s 387ms/step - loss: 0.4505 - acc: 0.8404 - val_loss: 0.5696 - val_acc: 0.8130

Epoch 37/50
97/97 [==============================] - 37s 385ms/step - loss: 0.4784 - acc: 0.8331 - val_loss: 0.8762 - val_acc: 0.7438

Epoch 38/50
97/97 [==============================] - 38s 388ms/step - loss: 0.5010 - acc: 0.8233 - val_loss: 0.6526 - val_acc: 0.7954

Epoch 39/50
97/97 [==============================] - 38s 388ms/step - loss: 0.5280 - acc: 0.8142 - val_loss: 1.0087 - val_acc: 0.7265

Epoch 40/50
97/97 [==============================] - 38s 389ms/step - loss: 0.5382 - acc: 0.8133 - val_loss: 1.6585 - val_acc: 0.6371

Epoch 41/50
97/97 [==============================] - 38s 390ms/step - loss: 0.5594 - acc: 0.8043 - val_loss: 1.9609 - val_acc: 0.6022

Epoch 42/50
97/97 [==============================] - 38s 388ms/step - loss: 0.5544 - acc: 0.8064 - val_loss: 1.2985 - val_acc: 0.6829

Epoch 43/50
97/97 [==============================] - 38s 387ms/step - loss: 0.5589 - acc: 0.8052 - val_loss: 1.0972 - val_acc: 0.7174

Epoch 44/50
97/97 [==============================] - 38s 387ms/step - loss: 0.5185 - acc: 0.8201 - val_loss: 0.7349 - val_acc: 0.7882

Epoch 45/50
97/97 [==============================] - 38s 388ms/step - loss: 0.4940 - acc: 0.8278 - val_loss: 0.8875 - val_acc: 0.7360

Epoch 46/50
97/97 [==============================] - 38s 388ms/step - loss: 0.4770 - acc: 0.8333 - val_loss: 0.6278 - val_acc: 0.8034

Epoch 47/50
97/97 [==============================] - 38s 387ms/step - loss: 0.4578 - acc: 0.8408 - val_loss: 0.7038 - val_acc: 0.7763

Epoch 48/50
97/97 [==============================] - 38s 390ms/step - loss: 0.4370 - acc: 0.8467 - val_loss: 0.5539 - val_acc: 0.8199

Epoch 49/50
97/97 [==============================] - 38s 388ms/step - loss: 0.4157 - acc: 0.8545 - val_loss: 0.4505 - val_acc: 0.8532

Epoch 50/50
97/97 [==============================] - 37s 386ms/step - loss: 0.3947 - acc: 0.8605 - val_loss: 0.4278 - val_acc: 0.8604

Model took 1907.49 seconds to train

![Image]()
Accuracy on test data is: 86.04

