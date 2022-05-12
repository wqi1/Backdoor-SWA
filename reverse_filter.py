import torch
import os
import csv
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torchvision
import random
import glob
import pytorch_ssim
import time

# TrojAI path
TrojanAI_dataset_dir = './test/'   # dataset directory
result_path = "./test.csv"  # file directory to save the result
example_path = "/clean_example_data/"  # example directory for round3
# example_path = "/example_data/"  # example directory for round2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

f = open(result_path, 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["model_id", "model_type", "step1_predict_target", "step2_success_rate", "step2_ssim",
                     "trojan_predict", "predict_target"])

random.seed(1)

#######################################
# Hyper Parameters
#######################################
ks = 2  # number of summation candidate
kd = 3  # number of divergence candidate

opt_sample_num = 10  # number of images for optimization
batch_size = 10
epoch_instagram = 101
gamma = 10
learning_rate = 0.1

# threshold to determine trojaned/clean
success_rate_threshold = 0.9
ssim_threshold = 0.3


#######################################
# Load Models and Detect
#######################################
def main():
    time_start = time.time()

    dirs = get_sub_dirs(TrojanAI_dataset_dir)
    for dir_ in dirs:
        trojan_flag = False

        model = torch.load(dir_ + '/model.pt')
        model.to(device)

        model_id = os.path.basename(dir_)
        print(model_id)
        model_type = model.__class__.__name__

        weight = extract_weight(model)
        class_num = weight.shape[0]

        # Summation
        weight_sum = np.zeros(class_num)
        for i in range(class_num):
            weight_sum[i] = sum(weight[i])
        weight_sum_max = np.argsort(-weight_sum)

        weight_sum_sus_target = []
        for i in range(ks):
            weight_sum_sus_target.append(weight_sum_max[i])

        # Divergence
        sim_matrix = calc_sim(weight)
        sim_mean = []
        for i in range(class_num):
            sim = sum(sim_matrix[i]) - 1
            sim = sim / (class_num - 1)
            sim_mean.append(sim)
        similarity_sus_target = calc_suspicious_target(sim_mean, kd)

        # Union of summation and divergence
        sus_target = []
        sus_target.append(weight_sum_sus_target[0])
        for i in range(kd):
            if similarity_sus_target[i] not in sus_target:
                sus_target.append(similarity_sus_target[i])
        for i in range(ks-1):
            if weight_sum_sus_target[i+1] not in sus_target:
                sus_target.append(weight_sum_sus_target[i+1])

        for t in range(len(sus_target)):  # consider each suspicious target
            current_target = sus_target[t]
            sus_victim = list(range(class_num))
            sus_victim.pop(current_target)

            success_rate_sum = 0
            ssim_list = []

            for v in range(len(sus_victim)):  # consider each victim
                current_victim = sus_victim[v]
                success_rate_temp, loss_ssim_temp = reverse_filter(model, current_target, current_victim, class_num, model_id)

                success_rate_sum = success_rate_sum + success_rate_temp
                ssim_list.append(loss_ssim_temp)

            # calculate the mean success rate of all classes
            success_rate_mean = success_rate_sum / len(sus_victim)

            # calculate the mean ssim for all successfully reversed classes
            if sum(ssim_list) != 0:
                ssim_mean = sum(ssim_list) / (len(ssim_list) - ssim_list.count(0))
                ssim_mean = ssim_mean/batch_size
            else:
                ssim_mean = 0

            # If success rate and ssim are both satisfied, we consider the model trojaned.
            if success_rate_mean > success_rate_threshold and ssim_mean > ssim_threshold:
                trojan_flag = True
                break

        csv_writer.writerow([model_id, model_type, sus_target, success_rate_temp, loss_ssim_temp, trojan_flag, current_target])

    time_end = time.time()
    csv_writer.writerow(["time:", time_end-time_start])


# load TrojAI model paths
def get_sub_dirs(root_path):
    root_depth = len(root_path.split(os.path.sep))
    c = []
    for root, dirs, files in os.walk(root_path, topdown=True):
        for name in dirs:
            dir_path = os.path.join(root, name)
            dir_depth = len(dir_path.split(os.path.sep))

            if dir_depth == root_depth:
                c.append(dir_path)
            else:
                break
    return c


#######################################
# Step1: Static Weight Analysis
#######################################
# Extract the weight of the last layer
def extract_weight(model):
    param = []
    for name, parameters in model.named_parameters():
        param.append(parameters.detach().cpu().numpy())
    weight_dense = param[-2]
    weight_dense = weight_dense.squeeze()
    # Normalize
    weight_max = np.max(abs(weight_dense))
    weight_dense = weight_dense/weight_max
    return weight_dense


# Calculate cosine similarity
def cosine_similarity(a, b):
    vector_a = np.mat(a)
    vector_b = np.mat(b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos


# Calculate similarity martix
def calc_sim(weight):
    sim_matrix = np.ones((len(weight), len(weight)), dtype=np.float64)
    for i in range(len(weight) - 1):
        for j in range(i + 1, len(weight)):
            sim = cosine_similarity(weight[i], weight[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim
    return sim_matrix


# Calculate top-k suspicious target
def calc_suspicious_target(sim, kk):
    sim_list = sim.copy()
    suspicious_target = np.argsort(np.array(sim_list))[0:kk]
    return suspicious_target


#######################################
# Step2: Trigger Reverse Engineering
#######################################
# Convert image to tensor
def img2tensor(imgdir):
    origin_img = Image.open(imgdir)
    crop_obj = torchvision.transforms.CenterCrop((224, 224))
    origin_img = crop_obj(origin_img)
    img_tensor = to_tensor(origin_img).to(device)
    input_img = torch.unsqueeze(img_tensor, 0)
    return input_img


# Generate data batch
def read_data(batch_size, features, lables):
    nums_example = len(features)
    indices = list(range(nums_example))
    random.shuffle(indices)
    for i in range(0, nums_example, batch_size):  # range(start, stop, step)
        index_tensor = torch.tensor(indices[i: min(i + batch_size, nums_example)])
        yield features[index_tensor], lables[index_tensor]


# Calculate the success rate of a data batch
def cal_success_rate(model, target_class, img_n_opt, out_length):
    new_index = 0
    success_list_1 = np.zeros(out_length)
    if out_length == 0:
        return 0

    for j in range(out_length):
        out_j = model(img_n_opt[j])
        out_j = torch.argmax(out_j, dim=1)
        new_index = new_index + 1
        if out_j == target_class:
            success_list_1[j] = 1
    success_rate = success_list_1.sum() / len(success_list_1)
    return success_rate


#######################################
# step 2-2：Instagram Filter
#######################################
# Apply filter trigger to image x
def apply_filter(filter, x):
    a = torch.ones([1, 1, 224, 224]).to(device)
    x_t = torch.mm(filter, torch.reshape(torch.cat((x, a, a), 1), (5, -1)))
    x_t_a = torch.stack((x_t[3], x_t[3], x_t[3]))
    x_t_rgb = torch.stack((x_t[0], x_t[1], x_t[2]))
    x_t_ = 1-x_t_a + x_t_rgb*x_t_a
    x_t_ = torch.reshape(x_t_, (-1, 3, 224, 224))
    x_t_tanh = torch.tanh(x_t_) / (2 - 1e-7) + 0.5
    return x_t_tanh


# Generate data batches for the whole optimization process
def data_generate_instagram(num, img_list, model_id, class_num):
    break_flag = 0
    # length = len(img_list)
    features = torch.zeros((num, 1, 3, 224, 224))
    labels = torch.zeros((num, 1))
    path_image_num = len(glob.glob(TrojanAI_dataset_dir + model_id + example_path + "*.png"))  # total image number
    number_example_images = int(path_image_num / class_num)  # image number for each class
    new_indices = 0
    for n in range(0, number_example_images):
        if break_flag == 1:
            break
        for i in img_list:
            IMG_PATH = TrojanAI_dataset_dir + model_id + example_path + "class_" + str(i) + '_example_' + str(n) + '.png'
            i_tensor = img2tensor(IMG_PATH)
            features[new_indices] = i_tensor
            labels[new_indices] = i
            new_indices = new_indices + 1
            if new_indices == num:
                break_flag = 1
                break
    return features, labels


# One step of optimization
def opt_train_filter(model, filter, features_ori, optimizer, criterion, target_class, img_n):
    optimizer.zero_grad()

    target = torch.tensor([target_class]).cuda()
    features_opt = torch.zeros_like(features_ori)
    img_n_opt = torch.zeros_like(img_n)

    for i in range(len(features_ori)):  # apply trigger for current batch
        features_opt[i] = apply_filter(filter, features_ori[i])

    for i in range(len(img_n)):  # apply trigger for all images
        img_n_opt[i] = apply_filter(filter, img_n[i])

    loss_1 = 0
    for i in range(len(features_opt)):    # sum the loss1 for all images in a batch
        features_opti = features_opt[i]
        out_opt = model(features_opti)
        loss_1_i = criterion(out_opt, target)
        loss_1 = loss_1 + loss_1_i

    loss_2 = 0
    for i in range(len(features_opt)):    # sum the loss2 for all images in a batch
        one_opti = features_opt[i]
        one_ori = features_ori[i]
        loss_2_i = pytorch_ssim.ssim(one_ori, one_opti)
        loss_2 = loss_2 + loss_2_i

    loss = loss_1 - gamma * loss_2

    loss = loss.to(device)
    loss.requires_grad_(True)

    loss.backward()
    optimizer.step()
    return loss, loss_1, loss_2, img_n_opt


# Reverse engineering process for filter attack
def reverse_filter(model, target_class, victim_class, class_num, model_id):
    print('target_class:', target_class)
    print('victim_class:', victim_class)

    img_list = [victim_class]  # only victim class needs optimization

    img_n, img_labels = data_generate_instagram(opt_sample_num, img_list, model_id, class_num)
    img_n = img_n.cuda()

    filter_ = torch.rand((4, 5)).to(device)
    filter_.requires_grad_(True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([filter_], lr=0.01)

    print("start training: ")

    sr_max = 0
    loss_ssim = 0
    sr_early_stop = True  # record if success rate is 0
    loss_train_2 = 0

    for i in range(epoch_instagram):
        epoch_sr_list = []
        for X, y in read_data(batch_size, img_n, img_labels):
            loss_train, loss_train_1, loss_train_2, img_n_opti = opt_train_filter(model, filter_, X, optimizer,
                                                                           criterion, target_class, img_n)

            sr = cal_success_rate(model, target_class, img_n_opti, opt_sample_num)
            epoch_sr_list.append(sr)

            print("epoch%d" % (i))
            print('success rate: %f, l1: %f, l2: %f' % (sr, float(loss_train_1), float(loss_train_2)))

        sr_mean = np.mean(epoch_sr_list)
        # If success rate is more than 0, update the flag.
        if sr_mean > 0:
            sr_early_stop = False
        # If success rate is always 0 after 50 epochs, early stop.
        if i >= 50 and sr_early_stop:
            break
        # Record the maximum success rate of the last 90 epochs and corresponding ssim.
        if i >= 90 and sr_mean > sr_max:
            sr_max = sr_mean
            loss_ssim = float(loss_train_2)

    return sr_max, loss_ssim


if __name__ == "__main__":
    main()
