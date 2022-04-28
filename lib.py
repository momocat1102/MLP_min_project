import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow import keras
import matplotlib.pyplot as plt

def set_generator(path):
    traindir = path + '/train'
    testdir = path + '/test'
    train_datagen = ImageDataGenerator(# rescale=1./255, 
                                        #zca_whitening = True,
                                        rotation_range=10,
                                        width_shift_range=0.15,
                                        height_shift_range=0.15,
                                        shear_range=0.10, 
                                        zoom_range= [0.5,1.1],
                                        horizontal_flip=True,
                                        fill_mode='constant', cval=0)

    train_generator = train_datagen.flow_from_directory(traindir,
                                            # classes=dic,
                                            # save_to_dir='./saveto',
                                            target_size=(224, 224),
                                            batch_size=8)

    test_datagen = ImageDataGenerator()# rescale=1./255)

    test_generator = test_datagen.flow_from_directory(testdir,
                                            # classes=dic,
                                            target_size=(224, 224),
                                            shuffle=False,
                                            batch_size=1)

    return train_generator, test_generator


def build_train_model(train_generator, test_generator, class_weight, classes, save_path):
    model = keras.applications.ResNet152V2(include_top=False,weights= 'imagenet', input_tensor=Input(shape=(224, 224, 3)))
    x = model.output
    x = tf.nn.dropout(x, 0.55)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
        
    # compile
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000064),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    estop = EarlyStopping(monitor='val_loss', patience=12, mode='auto', verbose=1)

    # lr reduce
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                            patience=5, mode='auto', verbose=1, cooldown=2,
                            min_lr=1e-6)

    modelcp = ModelCheckpoint(save_path, save_best_only=True, save_weights_only=True)

    # model.load_weights(save_path)
    train_history = model.fit(train_generator, batch_size=16,
            epochs=60, verbose=1, class_weight=class_weight,
            validation_data=test_generator, callbacks=[estop, reduce_lr, modelcp])
    
    # model.save_weights(save_path)

    return train_history

def show_train_history(train_history, train_acc, test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel(train_acc)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def build_model(classes, load_path):
    model = keras.applications.ResNet152V2(include_top=False,weights= 'imagenet', input_tensor=Input(shape=(224, 224, 3)))
    x = model.output
    x = tf.nn.dropout(x, 0.55)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(classes, activation='sigmoid')(x)
    model = Model(inputs=model.input, outputs=predictions)
        
    # compile
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000064),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    model.load_weights(load_path)

    return model