import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def set_generator(path):
    traindir = path + '/train'
    testdir = path + '/test'
    train_datagen = ImageDataGenerator( #zca_whitening = True,
                                        rotation_range=10,
                                        width_shift_range=0.25,
                                        height_shift_range=0.22,
                                        shear_range=0.30, 
                                        zoom_range= [0.5,1.1],
                                        horizontal_flip=True,
                                        fill_mode='constant', cval=0)

    train_generator = train_datagen.flow_from_directory(traindir,
                                            # classes=dic,
                                            target_size=(224, 224),
                                            batch_size=8)

    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(testdir,
                                            # classes=dic,
                                            target_size=(224, 224),
                                            shuffle=False,
                                            batch_size=1)

    return train_generator, test_generator


def build_model(classes, save_path = None):
    model = keras.applications.EfficientNetV2M(include_top=False,weights= 'imagenet', input_tensor=Input(shape=(224, 224, 3)))
    x = model.output
    x = tf.nn.dropout(x, 0.55)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
        
    # compile
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000064),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    if save_path != None:
        model.load_weights(save_path)
    return model

def show_train_history(train_history, train_acc, test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel(train_acc)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# 進行訓練
train_generator, test_generator = set_generator('./data_min_motorcycle')
print(test_generator.class_indices)
model = build_model(10)
# # {'BWS': 0, 'CYGNUS Gryphus': 1, 'CygnusX': 2, 'Force': 3, 'GP125': 4, 'KRV': 5, 'VJR': 6, 'jet_S_SR': 7, 'jet_sl': 8, 'racing _s': 9}
# class_weight = {
#         0: 2.,
#         1: 2.,
#         2: 2.,
#         3: 2.,
#         4: 2.,
#         5: 2.,
#         6: 2.,
#         7: 1.,
#         8: 2.,
#         9: 2.
# }

# estop = EarlyStopping(monitor='val_loss', patience=12, mode='auto', verbose=1)

# # lr reduce
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                         patience=5, mode='auto', verbose=1, cooldown=2,
#                         min_lr=1e-6)

# modelcp = ModelCheckpoint('one_model.h5', save_best_only=True, save_weights_only=True)

# train_history = model.fit(train_generator, batch_size=16,
#         epochs=100, verbose=1, validation_data=test_generator,
#         class_weight=class_weight, callbacks=[estop, reduce_lr, modelcp])

# show_train_history(train_history, 'accuracy','val_accuracy')
# show_train_history(train_history, 'loss','val_loss')


# 進行預測
path = './test_dataset'
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(path,
                                            # classes=dic,
                                            target_size=(224, 224),
                                            shuffle=False,
                                            batch_size=1)
model = build_model(10, './one_model.h5')
model.evaluate(test_generator, verbose=1)
dic_re = test_generator.class_indices
# {'BWS': 0, 'CYGNUS Gryphus': 1, 'CygnusX': 2, 'Force': 3, 'GP125': 4, 'KRV': 5, 'VJR': 6, 'jet_S_SR': 7, 'jet_sl': 8, 'racing _s': 9}
dic = dict(zip(dic_re.values(), dic_re.keys()))
print(dic)
preds = tf.argmax(model.predict(test_generator, verbose=1), axis=1).numpy().tolist()
moto_list = ['BWS', 'CYGNUS Gryphus', 'CygnusX', 'Force', 'GP125', 'jet_sl', 'jet_S_SR', 'KRV', 'racing _s', 'VJR']
finall_predict_true = list(0 for i in range(len(moto_list)))
finall_predict_false = list(0 for i in range(len(moto_list)))
count = 0
for _, labels in test_generator:
    if count > 43:
        break
    # print(labels, labels.shape)
    for i in range(len(labels[0])):
        if labels[0][i] == 1.:
            label = i 
    if preds[count] == label:
        for j in range(len(moto_list)):
            if moto_list[j] == dic[label]:
                finall_predict_true[j] += 1
    else:
        for i in range(len(moto_list)):
            if moto_list[i] == dic[label]:
                finall_predict_false[i] += 1
    count += 1

plt.figure(figsize=(15, 6))
X = np.arange(len(moto_list))
Y1 = finall_predict_true
Y2 = finall_predict_false
print(Y1)
plt.bar(X, Y1, alpha=0.9, width=0.3, facecolor='blue', edgecolor='white', label='Predict_true', lw=1)
plt.bar(X + 0.35, Y2, alpha=0.9, width=0.3, facecolor='red', edgecolor='white', label='Predict_false', lw=1)
plt.xticks(X + 0.35 / 2, moto_list)
plt.title("Predict")
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.show()