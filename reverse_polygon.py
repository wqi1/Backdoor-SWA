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
import time

# TrojAI path
TrojanAI_dataset_dir = './TrojanAI/round3/'  # dataset directory
result_path = "./test.csv"  # file directory to save the result
example_path = "/clean_example_data/"  # example directory for round3
# example_path = "/example_data/"  # example directory for round2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

f = open(result_path, 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["model_id", "model_type", "poison_type", "step1_predict_target", "step2_success_rate",
                     "step2_mask_size", "step2_grid_num", "step2_activation", "trojan_predict", "predict_target", "predict_victim"])

random.seed(1)

#######################################
# Hyper Parameters
#######################################
ks = 2  # number of summation candidate
kd = 3  # number of divergence candidate
kv = 2  # number of victim candidate
universal_threshold = 0.1  # threshold to determine universal/specific attack

opt_sample_num_universal = 40  # number of images for optimization
opt_sample_num_specific = 10
epoch_polygon = 201
batch_size = 10
# beta = 0.001
learning_rate = 0.1

# threshold to determine trojaned/clean
mask_size_universal = 1500
mask_size_specific = 500
grid_num_threshold = 100
target_jaccard_threshold = 0.5

# neuron activation analysis parameters
clean_img_num = 10
set_len = 30  # number of top neuron activation for all images
k = 20  # number of top neuron activation for each image

# cost adjustment and early stop parameters
init_cost = 0.001  # beta
attack_succ_threshold = 0.99
patience = 5
early_stop_patience = 10
early_stop = True
# cost_multiplier_up = 1.5
# cost_multiplier_down = 1.5


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

        # Determine universal or specific
        poison_type = "specific"
        sim_sort = np.sort(np.array(sim_mean))
        diff = sim_sort[1] - sim_sort[0]
        if diff > universal_threshold:
            poison_type = "universal"

        # Deal with different model structure
        modules = list(model.children())
        if model_type == "SqueezeNet":
            fc = modules[-1][-3]
            layer_size = fc.in_channels
            class_num = fc.out_channels
            fc.register_forward_hook(hook_forward_fn)
        elif model_type == "MobileNetV2" or model_type == "VGG":
            fc = modules[-1][-1]
            layer_size = fc.in_features
            class_num = fc.out_features
            fc.register_forward_hook(hook_forward_fn)
        else:
            fc = modules[-1]
            layer_size = fc.in_features
            class_num = fc.out_features
            fc.register_forward_hook(hook_forward_fn)

        # For universal attack
        if poison_type == "universal":
            current_target = similarity_sus_target[0]  # only need to consider the first Divergence suspicious target
            success_rate_temp, loss_mask_temp, trojan_flag = reverse_trigger_universal(model, current_target, class_num, model_id)
            csv_writer.writerow([model_id, model_type, poison_type, sus_target, success_rate_temp,
                 loss_mask_temp, "", trojan_flag, current_target, "all"])
        # For label-specific attack
        else:
            for t in range(len(sus_target)):  # consider each suspicious target
                current_target = sus_target[t]
                current_target_sim = sim_matrix[current_target]
                sus_victim = np.argsort(-np.array(current_target_sim))[1:kv+1]  # calculate suspicious victim candidates
                for v in range(len(sus_victim)):  # consider each target-victim pair
                    current_victim = sus_victim[v]
                    success_rate_temp, loss_mask_temp, grid_num, activation_jaccard, trojan_flag = reverse_trigger_specific(model, current_target, current_victim, class_num, model_id)
                # If a model has been considered trojaned, we don't need to try other target-victim pair.
                    if trojan_flag:
                        break
                if trojan_flag:
                    break
            csv_writer.writerow([model_id, model_type, poison_type, sus_target, success_rate_temp,
                loss_mask_temp, grid_num, activation_jaccard, trojan_flag, current_target, current_victim])

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
# step 2-1：Polygon
#######################################
# Stamp the trigger to image x
def apply_trigger(mask, trigger, x):
    x_t = x * (1 - mask) + trigger * mask
    return x_t


def arctanh(x):
    x_arctanh = 0.5 * (torch.log((1+x) / (1-x)))
    return x_arctanh


# Generate data batches for the whole optimization process
def data_generate_polygon(num, img_list, model_id, class_num):
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
def opt_train(model, trigger, mask, features_ori, optimizer, criterion, target_class, img_n, cost):
    optimizer.zero_grad()

    trigger_tanh = torch.tanh(trigger) / (2 - 1e-7) + 0.5
    mask_tanh = torch.tanh(mask) / (2 - 1e-7) + 0.5
    target = torch.tensor([target_class]).cuda()
    features_opt = apply_trigger(mask_tanh, trigger_tanh, features_ori)  # apply trigger for current batch
    img_n_opt = apply_trigger(mask_tanh, trigger_tanh, img_n)  # apply trigger for all images

    loss_1 = 0
    for i in range(len(features_opt)):    # sum the loss for all images in a batch
        features_opti = features_opt[i]
        out_opt = model(features_opti)
        loss_1_i = criterion(out_opt, target)
        loss_1 = loss_1 + loss_1_i

    loss_2 = torch.norm(mask_tanh, p=1)

    loss = loss_1 + cost * loss_2

    loss = loss.to(device)
    loss.requires_grad_(True)

    loss.backward()
    optimizer.step()
    return loss, loss_1, loss_2, img_n_opt, mask_tanh


# Reverse engineering process for universal attack
def reverse_trigger_universal(model, target_class, class_num, model_id):
    print(target_class)

    trojan_flag = False  # record if a model is trojaned

    img_list = list(range(0, class_num))  # creat a list for image classes that need to be optimized
    img_list.pop(target_class)

    img_n, img_labels = data_generate_polygon(opt_sample_num_universal, img_list, model_id, class_num)
    img_n = img_n.cuda()

    trigger_ = torch.rand(1, 3, 224, 224).to(device)
    mask_ = torch.rand(1, 224, 224).to(device) * 0.001
    trigger_ = torch.clamp(trigger_, min=0, max=1)
    mask_ = torch.clamp(mask_, min=0, max=1)
    trigger_arctan = arctanh((trigger_ - 0.5) * (2 - 1e-7))
    mask_arctan = arctanh((mask_ - 0.5) * (2 - 1e-7))
    trigger_arctan.requires_grad_(True)
    mask_arctan.requires_grad_(True)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([trigger_arctan, mask_arctan], lr=learning_rate, betas=(0.5, 0.9))
    print("start training: ")

    sr_max = 0

    # cost_set_counter = 0
    # cost_up_counter = 0
    # cost_down_counter = 0
    # cost_up_flag = False
    # cost_down_flag = False

    early_stop_counter = 0
    early_stop_mask_best = float("inf")
    cost = init_cost

    # Optimizaiton process for each epoch
    for i in range(epoch_polygon):
        epoch_sr_list = []
        for X, y in read_data(batch_size, img_n, img_labels):
            loss_train, loss_train_1, loss_train_2, img_n_opti, mask_tanh = opt_train(model, trigger_arctan, mask_arctan, X, optimizer,
                                                                           criterion, target_class, img_n, cost)

            sr = cal_success_rate(model, target_class, img_n_opti, opt_sample_num_universal)
            epoch_sr_list.append(sr)
            print("epoch%d" % (i))
            print('success rate: %f, l1: %f, l2: %f' % (sr, float(loss_train_1), float(loss_train_2)))
        sr_mean = np.mean(epoch_sr_list)

        # If the size and success rate requirements are all satisfied, the model is trojaned.
        if float(loss_train_2) < mask_size_universal and sr_mean > attack_succ_threshold:
            trojan_flag = True
            break

        # Start using early stop from epoch 50
        if i > 49 and early_stop:
            if float(loss_train_2) < float('inf'):
                if float(loss_train_2) >= early_stop_mask_best:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
            early_stop_mask_best = min(float(loss_train_2), early_stop_mask_best)

            if(early_stop_counter >= early_stop_patience and sr >= attack_succ_threshold):
                print('early stop')
                # For universal attack, if the mask size of generated trigger is small enough, we consider it trojaned.
                if float(loss_train_2) < mask_size_universal:
                    trojan_flag = True
                break

        # Adjust beta
        # if cost == 0 and sr_mean >= attack_succ_threshold:
        #     cost_set_counter += 1
        #     if cost_set_counter >= 2:
        #         cost = init_cost
        #         cost_up_counter = 0
        #         cost_down_counter = 0
        #         cost_up_flag = False
        #         cost_down_flag = False
        # else:
        #     cost_set_counter = 0
        #
        # if sr_mean >= attack_succ_threshold:  # satisfy the success rate requirment, cost_up+1
        #     cost_up_counter += 1
        #     cost_down_counter = 0
        # else:  # otherwise, cost_down+1
        #     cost_up_counter = 0
        #     cost_down_counter += 1
        #
        # if cost_up_counter >= patience:  # reach cost_up patience, increase the cost
        #     cost_up_counter = 0
        #     cost *= cost_multiplier_up
        #     cost_up_flag = True
        # elif cost_down_counter >= patience:  # reach cost_down patience, decrease the cost
        #     cost_down_counter = 0
        #     cost /= cost_multiplier_down
        #     cost_down_flag = True

    return sr_max, loss_train_2, trojan_flag


# Reverse engineering process for label-specific attack
def reverse_trigger_specific(model, target_class, victim_class, class_num, model_id):
    print(target_class)
    print(victim_class)

    trojan_flag = False  # record if a model is trojaned

    img_list = [victim_class]  # only victim class needs optimization

    img_n, img_labels = data_generate_polygon(opt_sample_num_specific, img_list, model_id, class_num)
    img_n = img_n.cuda()

    trigger_ = torch.rand(1, 3, 1, 1).to(device)
    mask_ = torch.rand(1, 224, 224).to(device) * 0.001
    mask_[:, 112 - 25:112 + 25, 112 - 25:112 + 25] = 0.99
    trigger_ = torch.clamp(trigger_, min=0, max=1)
    mask_ = torch.clamp(mask_, min=0, max=1)
    trigger_arctan = arctanh((trigger_ - 0.5) * (2 - 1e-7))
    mask_arctan = arctanh((mask_ - 0.5) * (2 - 1e-7))
    trigger_arctan.requires_grad_(True)
    mask_arctan.requires_grad_(True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([trigger_arctan, mask_arctan], lr=learning_rate, betas=(0.5, 0.9))

    print("start training: ")

    sr_max = 0
    grid_num = 0
    target_jaccard = 0

    # cost_set_counter = 0
    # cost_up_counter = 0
    # cost_down_counter = 0
    # cost_up_flag = False
    # cost_down_flag = False

    early_stop_counter = 0
    early_stop_mask_best = float("inf")
    cost = init_cost

    # Optimizaiton process for each epoch
    for i in range(epoch_polygon):
        epoch_sr_list = []
        for X, y in read_data(batch_size, img_n, img_labels):
            loss_train, loss_train_1, loss_train_2, img_n_opti, mask_tanh = opt_train(model, trigger_arctan, mask_arctan, X, optimizer,
                                                                           criterion, target_class, img_n, cost)

            sr = cal_success_rate(model, target_class, img_n_opti, opt_sample_num_specific)

            epoch_sr_list.append(sr)
            print("epoch%d" % (i))
            print('success rate: %f, l1: %f, l2: %f' % (sr, float(loss_train_1), float(loss_train_2)))
        sr_mean = np.mean(epoch_sr_list)

        # If the size and success rate requirements are all satisfied, analyze the generated trigger.
        if float(loss_train_2) < mask_size_specific and sr_mean > attack_succ_threshold:
            # Excluding Adversarial Perturbations
            mask_tanh_copy = mask_tanh.clone().detach()
            mask_np = mask_tanh_copy.cpu().detach().numpy().squeeze()

            grid_num = calc_grid_num(mask_np)
            if grid_num < grid_num_threshold:
                trojan_flag = True
            else:
                break

            # Excluding Natural features
            neuron_set_target = neuron_set_generate(model, model_id, target_class, clean_img_num, k,
                                                    set_len)  # activation set of clean images belonging to the target label
            for n in range(10):
                trigger_img = img_n_opti[n]
                opt_neuron_set = activation_topk_neurons(trigger_img, model,
                                                         set_len)  # activation set of triggered images
                target_jaccard_i = jaccard_index(opt_neuron_set, neuron_set_target)
                target_jaccard = target_jaccard + target_jaccard_i
            target_jaccard = target_jaccard / 10

            if target_jaccard > target_jaccard_threshold:
                trojan_flag = False
            break

        # Start using early stop from epoch 50
        if i > 49 and early_stop:
            if float(loss_train_2) < float('inf'):
                if float(loss_train_2) >= early_stop_mask_best:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
            early_stop_mask_best = min(float(loss_train_2), early_stop_mask_best)

            if(early_stop_counter >= early_stop_patience and sr_mean >= attack_succ_threshold):
                print('early stop')
                if float(loss_train_2) > mask_size_specific:
                    break
                # For specific attack, if the generated trigger is small enough, we further analysis the trigger to suppress false positives.
                else:
                    # Excluding Adversarial Perturbations
                    mask_tanh_copy = mask_tanh.clone().detach()
                    mask_np = mask_tanh_copy.cpu().detach().numpy().squeeze()

                    grid_num = calc_grid_num(mask_np)
                    if grid_num < grid_num_threshold:
                        trojan_flag = True
                    else:
                        break

                    # Excluding Natural features
                    neuron_set_target = neuron_set_generate(model, model_id, target_class, clean_img_num, k, set_len)  # activation set of clean images belonging to the target label
                    for n in range(10):
                        trigger_img = img_n_opti[n]
                        opt_neuron_set = activation_topk_neurons(trigger_img, model, set_len)  # activation set of triggered images
                        target_jaccard_i = jaccard_index(opt_neuron_set, neuron_set_target)
                        target_jaccard = target_jaccard + target_jaccard_i
                    target_jaccard = target_jaccard / 10

                    if target_jaccard > target_jaccard_threshold:
                        trojan_flag = False
                    break

        # Adjust beta
        # if cost == 0 and sr_mean >= attack_succ_threshold:
        #     cost_set_counter += 1
        #     if cost_set_counter >= 2:
        #         cost = init_cost
        #         cost_up_counter = 0
        #         cost_down_counter = 0
        #         cost_up_flag = False
        #         cost_down_flag = False
        # else:
        #     cost_set_counter = 0
        #
        # if sr_mean >= attack_succ_threshold:  # satisfy the success rate requirment, cost_up+1
        #     cost_up_counter += 1
        #     cost_down_counter = 0
        # else:  # otherwise, cost_down+1
        #     cost_up_counter = 0
        #     cost_down_counter += 1
        #
        # if cost_up_counter >= patience:  # reach cost_up patience, increase the cost
        #     cost_up_counter = 0
        #     cost *= cost_multiplier_up
        #     cost_up_flag = True
        # elif cost_down_counter >= patience:  # reach cost_down patience, decrease the cost
        #     cost_down_counter = 0
        #     cost /= cost_multiplier_down
        #     cost_down_flag = True

    return sr_max, loss_train_2, grid_num, target_jaccard, trojan_flag


#######################################
# Step3: Trigger Analysis
#######################################
# Calculate the grid number a trigger covers
def calc_grid_num(mask):
    grid_size = 7
    grid_list = []
    for x in range(224):
        for y in range(224):
            if mask[x][y] > 0.5:
                grid_x = x // grid_size
                grid_y = y // grid_size
                if [grid_x, grid_y] in grid_list:
                    pass
                else:
                    grid_list.append([grid_x, grid_y])
    grid_num = len(grid_list)
    return grid_num


# Calculate index of the top-k activated neuron for all images
def neuron_set_generate(model, model_id, label, img_num, k, set_num):
    neuron_set = []
    count = 0
    break_flag = 0
    for i in range(img_num):
        if break_flag == 1:
            break
        img_path = TrojanAI_dataset_dir + model_id + example_path + 'class_' + str(label) + '_example_' + str(i) + ".png"
        img_tensor = img2tensor(img_path)
        topk_neurons_i = activation_topk_neurons(img_tensor, model, k)
        for neu in topk_neurons_i:
            if neu not in neuron_set:
                neuron_set.append(neu)
                count = count + 1
                if count == set_num:
                    break_flag = 1
                    break
    return neuron_set


# Calculate index of the top-k activated neuron for one image
def activation_topk_neurons(img_tensor, model, k):
    predict(model, img_tensor)
    neuron_activation = activ
    sort, index = torch.sort(-neuron_activation)
    topk_neurons = index.squeeze(0).tolist()[0:k]
    return topk_neurons


def predict(mynet, x):
    out_x = mynet(x)


def hook_forward_fn(module, input, output):
    global activ
    activ = input[0]
    return None


# Calculate Neuron Behavior Similarity
def jaccard_index(list_a, list_b):
    if list_a == list_b == []:
        return 1
    list_c = []
    for item in list_a:
        if item in list_b:
            list_c.append(item)
    list_c_1 = []
    for i in list_c:
        if not i in list_c_1:
            list_c_1.append(i)

    list_d = list_a + list_b
    list_d_1 = []
    for i in list_d:
        if not i in list_d_1:
            list_d_1.append(i)

    return len(list_c_1) / len(list_d_1)


if __name__ == "__main__":
    main()
