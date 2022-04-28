import os
import shutil

# 將二分類的model，其資料集建出來
def mkmotodir(moto_name, model_name, group_num):
    lis = ['test', 'train']
    for mode in lis:
        allmotoList  = os.listdir('./data_min_motorcycle/' + mode)
        for i in range(group_num):
            if allmotoList[i] != moto_name:
                class_name = '0' 
            else:
                class_name = '1' 
            imglist = os.listdir('./data_min_motorcycle/' + mode +'/' + allmotoList[i])
            if class_name == '1':
                num = len(imglist)
            else:
                if mode == 'train':
                    if moto_name == 'jet_S_SR':
                        num = 16
                    else:   
                        num = 8
                else:
                    if moto_name == 'jet_S_SR':
                        num = 4
                    else:   
                        num = 2

            for j in range(num):
                shutil.copy('./data_min_motorcycle/' + mode + '/' + allmotoList[i] + '/' + imglist[j], model_name + '/' + mode + '/' + class_name + '/' + imglist[j])

# 將所有資料建成一個資料集
def make_alltype_moto_dir(path):
    os.mkdir("./data_min_motorcycle/test")
    os.mkdir("./data_min_motorcycle/train")
    lis = os.listdir(path)
    for i in range(len(lis)):
        moto_imgs = os.listdir(path + lis[i])
        os.mkdir("./data_min_motorcycle/test/" + lis[i])
        os.mkdir("./data_min_motorcycle/train/" + lis[i])
        count = 0
        for j in range(len(moto_imgs)):
            if lis[i] == 'jet_S_SR':
                if count > 35:
                    mode = '/train/'
                else:
                    mode = '/test/'
            else:
                if count > 17:
                    mode = '/train/'
                else:
                    mode = '/test/'
            count += 1
            shutil.copy(path + lis[i] + '/' + moto_imgs[j], './data_min_motorcycle/' + mode + '/' + lis[i] + '/' + moto_imgs[j])

# 將測資的檔名加上label
def set_test_img_to_dir(path, path_to):
    lis = os.listdir(path)
    try:
        shutil.rmtree(path_to)
    except:
        pass
    os.mkdir(path_to)
    for i in range(len(lis)):
        moto_imgs = os.listdir(path + '/' + lis[i])
        for j in range(len(moto_imgs)):
            shutil.copyfile(path + '/' + lis[i] + '/' + moto_imgs[j], path_to + '/' + str(i) + '_' + lis[i] + str(j) + '.jpg')

# -----------------------------------------------------------------------------------------------------------
# make_alltype_moto_dir('./img') # 將10分類的資料集建出

# 將二分類的九個資料集建出
# lis = os.listdir('./data_min_motorcycle/train')
# print(lis)
# for i in range(len(lis)):
#     try:
#         shutil.rmtree('./model' + str(i))
#     except:
#         pass
#     try:
#         os.mkdir('./model' + str(i))
#         os.mkdir('./model' + str(i) + '/test')
#         os.mkdir('./model' + str(i) + '/train')
#         os.mkdir('./model' + str(i) + '/test/0')
#         os.mkdir('./model' + str(i) + '/test/1')
#         os.mkdir('./model' + str(i) + '/train/0')
#         os.mkdir('./model' + str(i) + '/train/1')
#     except:
#         pass

# for i in range(len(lis)):
#     mkmotodir(lis[i], './model' + str(i), len(lis))


# 將測試資料集建出
set_test_img_to_dir('./test_dataset', 'test_imgs')