import os
import cv2
import random
import math

data_path = "F:/University/Semester 3/AI/Lab/13/dataset/dataset"
img_size = (50,50)
class_org_map = {}

for i in os.listdir(data_path):
    folder_path = os.path.join(data_path, i)
    if (os.path.isdir(folder_path)):
        class_org_map[i] = []
        for j in os.listdir(folder_path):
            path = os.path.join(folder_path, j)
            image = cv2.imread(path, 0)
            if (image is not None):
                image = cv2.resize(image, img_size)
                img_v = image.flatten().tolist()
                class_org_map[i].append(img_v)

train_X = []
train_y = []
test_X = []
test_y = []

for i in class_org_map.items():
    class_name = i[0]
    images = i[1]
    random.shuffle(images)
    split_idx = int(0.7 * len(images))
    for j in range(0,split_idx):
        train_X.append(images[j])
        train_y.append(class_name)
    for j in range(split_idx, len(images)):
        test_X.append(images[j])
        test_y.append(class_name)

res = {}
k_arr = [3, 5, 7, 9]
for num in k_arr:
    euc = 0
    cos = 0
    for i in range(0,len(test_X)):
        euc_dists = []
        cos_dists = []
        for j in range(0,len(train_X)):
            sum_sq = 0
            dot = 0
            norm_a = 0
            norm_b = 0
            for k in range(0,len(test_X[i])):
                diff = test_X[i][k] - train_X[j][k]
                sum_sq += diff * diff
                dot += test_X[i][k] * train_X[j][k]
                norm_a += test_X[i][k] * test_X[i][k]
                norm_b += train_X[j][k] * train_X[j][k]
            euc_dists.append((math.sqrt(sum_sq), train_y[j]))
            if (norm_a > 0 and norm_b > 0):
                cos_dists.append((dot / (math.sqrt(norm_a) * math.sqrt(norm_b)), train_y[j]))
            else:
                cos_dists.append((0, train_y[j]))
        euc_dists.sort()
        cos_dists.sort(reverse=True)
        votes_euc = {}
        votes_cos = {}
        for j in range(0,num):
            label_euc = euc_dists[j][1]
            label_cos = cos_dists[j][1]
            votes_euc[label_euc] = votes_euc.get(label_euc, 0) + 1
            votes_cos[label_cos] = votes_cos.get(label_cos, 0) + 1
        pred_euc = max(votes_euc, key=votes_euc.get)
        pred_cos = max(votes_cos, key=votes_cos.get)
        if (pred_euc == test_y[i]):
            euc += 1
        if (pred_cos == test_y[i]):
            cos += 1
    if (len(test_X) > 0):
        acc_euc = (euc / len(test_X)) * 100
        acc_cos = (cos / len(test_X)) * 100
    else:
        acc_euc = 0
        acc_cos = 0
    res[num] = (acc_euc, acc_cos)

for num in k_arr:
    print("K=", num, ": Euclidean ", round(res[num][0], 2), "%, Cosine ", round(res[num][1], 2), "%")
    print("\n")