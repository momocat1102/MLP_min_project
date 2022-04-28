import tensorflow as tf
from lib import set_generator, build_model
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

path = './test_imgs'
lis = os.listdir(path)
moto_class = os.listdir('./data_min_motorcycle/test')
preds_label = list()
labels_list = list()
all_img = 0
pre_true = 0

for img in lis:
    label = img.split('.')[0].split('_')[0]
    labels_list.append(int(label))
    test_img = cv2.imread(path + '/' + img, cv2.IMREAD_COLOR)
    test_img = cv2.resize(test_img, (224, 224))
    test_img = test_img.reshape(1, 224, 224, 3).astype('float32') 
    test_img = test_img / 255.
    all_img += 1
    pred_list = [i for i in range(len(moto_class))]

    for i in range(len(moto_class)):
        # print('model' + str(i) + ':')
        model = build_model(2, './weight/model' + str(i) +'.h5')
        pred = model.predict(test_img, verbose=0).tolist()[0][1]
        pred_list[i] = pred

    print('pred_list:', pred_list)
    max = 0
    for i in range(len(pred_list)):
        if pred_list[max] < pred_list[i]:
            max = i

    print(moto_class[max], img)
    preds_label.append(max)

finall_predict_true = list(0 for i in range(len(moto_class)))
finall_predict_false = list(0 for i in range(len(moto_class)))

for i in range(len(preds_label)):
    if preds_label[i] == labels_list[i]:
        pre_true += 1
        finall_predict_true[preds_label[i]] += 1
    else:
        finall_predict_false[labels_list[i]] += 1

print(pre_true/all_img)
plt.figure(figsize=(15, 6))
moto_list = ['BWS', 'CYGNUS_Gryphus', 'CygnusX', 'Force', 'GP125', 'jet_sl', 'jet_S_SR', 'KRV', 'racing _s', 'VJR']
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


# for i in range(10):
#     train_generator, test_generator = set_generator('./model' + str(i))
#     print('model' + str(i) + ':')
#     model = build_model(2, './weight/model' + str(i) + '.h5')
#     model.evaluate(test_generator, verbose=1, batch_size=1)
#     pred = tf.argmax(model.predict(test_generator, verbose=1), axis=1).numpy().tolist()
#     # pred = model.predict(test_generator, verbose=1).tolist()
#     print(pred)
