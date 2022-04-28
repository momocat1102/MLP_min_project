from lib import set_generator, build_train_model, show_train_history
import os

moto_class = os.listdir('./data_min_motorcycle/train')
# 建立10個model
for i in range(len(moto_class)):
    train_generator, test_generator = set_generator('./model' + str(i))
    img_0_ = len(os.listdir('./model' + str(i) + '/train/0'))
    img_1_ = len(os.listdir('./model' + str(i) + '/train/1'))
    print('model' + str(i) + ':')
    class_weight = {
        0: img_1_ / img_0_,
        1: 1.
    }
    train_history = build_train_model(train_generator, test_generator, class_weight, 2, './weight/model' + str(i) + '.h5')

    show_train_history(train_history, 'accuracy','val_accuracy')
    show_train_history(train_history, 'loss','val_loss')


# ['BWS', 'CYGNUS Gryphus', 'CygnusX', 'Force', 'GP125', 'jet_sl', 'jet_S_SR', 'KRV', 'racing _s', 'VJR']
