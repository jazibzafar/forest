##
import torch
from src.checkpoints import load_dino_checkpoint, prepare_arch, load_finetuned_checkpoint
from tifffile import imread
from src.data_and_transforms import CenterCrop
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def dict_to_yml(file_to_write, dict_to_write):
    with open(file_to_write, 'w') as outfile:
        yaml.dump(dict_to_write, outfile, default_flow_style=False)


# convert the nd arrays to list because otherwise they get saved as binaries
def dict_array_to_list(dict_in):
    for key in dict_in.keys():
        dict_in[key] = dict_in[key].tolist()
    return dict_in


def euclidean_dist(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def df_dist(df_in):
    columns = df_in.columns
    index = columns.copy()
    size = len(columns)
    df_out = pd.DataFrame(np.zeros((size, size)), index=index, columns=columns)
    for i in df_in.columns:
        for j in df_in.columns:
            df_out[i][j] = euclidean_dist(df_in[i], df_in[j])
    return df_out


def evaluate_model_on_class(model):

    path = "/home/jazib/projects/data/species_classification/"
    classes = os.listdir(path)
    transform = CenterCrop(input_size=96)
    print(classes)

    model.eval()
    model.to('cuda')

    att_dict = {}
    out_dict = {}
    for cls in classes:
        folder = os.path.join(path, cls)
        image_list = os.listdir(folder)
        len_list = len(image_list)
        cls_att = np.zeros((37, 37))
        cls_out = np.zeros(384)
        for image_name in image_list:
            img_path = os.path.join(folder, image_name)
            img = imread(img_path)
            img = transform(img)
            img = img.unsqueeze(0).to('cuda')
            out = model(img)
            cls_out += out.squeeze(0).cpu().detach().numpy()
            att = model.get_last_selfattention(img)
            att = att.squeeze(0).cpu().detach().numpy()
            att = np.sum(att, axis=0) / 6
            cls_att += att
        att_dict[cls] = cls_att / len_list
        out_dict[cls] = cls_out / len_list
    return att_dict, out_dict


arch = "vit_small"
patch_size = 16
checkpoint_key = "teacher"
checkpoint_path = "/home/jazib/projects/savedmodels/dino_vit-s_f_500k.ckpt"
checkpoint = load_dino_checkpoint(checkpoint_path, checkpoint_key)
model_untuned = prepare_arch(arch, checkpoint, patch_size)



checkpoint_ft_path = "/home/jazib/projects/savedmodels/vit-s_f_500k_ds_class.ckpt"
ckpt_ft = load_finetuned_checkpoint(checkpoint_ft_path)
model_ft = prepare_arch(arch, ckpt_ft, patch_size)


##

d_att_un, d_out_un = evaluate_model_on_class(model_untuned)
d_att_ft, d_out_ft = evaluate_model_on_class(model_ft)

d_att_un_ls = dict_array_to_list(d_att_un)
d_out_un_ls = dict_array_to_list(d_out_un)
d_att_ft_ls = dict_array_to_list(d_att_ft)
d_out_ft_ls = dict_array_to_list(d_out_ft)


write_att_un = "./stats/vit-s_f_500k_pretrain_att.yml"
dict_to_yml(write_att_un, d_att_un_ls)

write_out_un = "./stats/vit-s_f_500k_pretrain_out.yml"
dict_to_yml(write_out_un, d_out_un_ls)

write_att_ft = "./stats/vit-s_f_500k_finetune_att.yml"
dict_to_yml(write_att_ft, d_att_ft_ls)

write_out_ft = "./stats/vit-s_f_500k_finetune_out.yml"
dict_to_yml(write_out_ft, d_out_ft_ls)
##
df_out_un = pd.DataFrame.from_dict(d_out_un)
df_out_ft = pd.DataFrame.from_dict(d_out_ft)
df_diff = df_out_un - df_out_ft
##
dist_un = df_dist(df_out_un)
dist_ft = df_dist(df_out_ft)
dist_di = df_dist(df_diff)


mask = np.triu(np.ones_like(dist_un, dtype=bool))
f1 = plt.figure(figsize=(11, 9))

f1.add_subplot(131)
sns.heatmap(dist_un, cmap='flare', mask=mask, square=True, linewidths=.5)
plt.title("untuned")
f1.add_subplot(132)
sns.heatmap(dist_ft, cmap='flare', mask=mask, square=True, linewidths=.5)
plt.title("finetuned")
f1.add_subplot(133)
sns.heatmap(dist_di, cmap='flare', mask=mask, square=True, linewidths=.5)
plt.title("diff")
plt.show()
f1.savefig("./figures/comparison_out_features.png")

##
path = "/home/jazib/projects/data/species_classification/"
classes = os.listdir(path)

f2 = plt.figure(figsize=(11, 9))
rows = 4
cols = 3
for i in range(len(classes)):
    key = classes[i]
    attention = d_att_un[key]
    f2.add_subplot(rows, cols, i+1)
    plt.imshow(attention)
    plt.axis('off')
    plt.title(key)
plt.show()
f2.savefig("./figures/pt_attentions.png")
##
f3 = plt.figure(figsize=(11, 9))
rows = 4
cols = 3
for i in range(len(classes)):
    key = classes[i]
    attention = d_att_ft[key]
    f3.add_subplot(rows, cols, i+1)
    plt.imshow(attention)
    plt.axis('off')
    plt.title(key)
plt.show()
f3.savefig("./figures/ft_attentions.png")

##
diff_att = {}
for key in d_att_un.keys():
    diff_att[key] = np.array(d_att_un[key]) - np.array(d_att_ft[key])

f4 = plt.figure(figsize=(11, 9))
rows = 4
cols = 3
for i in range(len(classes)):
    key = classes[i]
    attention = diff_att[key]
    f4.add_subplot(rows, cols, i+1)
    plt.imshow(attention)
    plt.axis('off')
    plt.title(key)
plt.show()
f4.savefig("./figures/di_attentions.png")



##
# img_ei_path = "/home/jazib/projects/data/species_classification/Ei/tree_259418_Ei.tif"
# img_ei = imread(img_ei_path)
# transform = CenterCrop(input_size=96)
# img_ei = transform(img_ei)
# img_ei = img_ei.unsqueeze(0)
#
# img_dgl_path = "/home/jazib/projects/data/species_classification/Dgl/tree_228995_Dgl.tif"
# img_dgl = imread(img_dgl_path)
# img_dgl = transform(img_dgl)
# img_dgl = img_dgl.unsqueeze(0)

# out_ei = model(img_ei)
# att_ei = model.get_last_selfattention(img_ei)
# att_ei = att_ei.squeeze(0)
# att_ei = att_ei.detach().numpy()
#
# out_dgl = model(img_dgl)
# att_dgl = model.get_last_selfattention(img_dgl)
# att_dgl = att_dgl.squeeze(0)
# att_dgl = att_dgl.detach().numpy()


# att_sum_ei = np.sum(att_ei, axis=0)
# plt.imshow(att_sum_ei)
# plt.show()
#
# att_sum_dgl = np.sum(att_dgl, axis=0)
# plt.imshow(att_sum_dgl)
# plt.show()


# np_ei = out_ei.detach().numpy()
# np_dgl = out_dgl.detach().numpy()
#
# print(euclidean_dist(np_ei, np_dgl))



