"""
General utils for csgo training data
Author: Bo
Created on 2022/12/02
Comments: 网上下载了别人的数据集，图片标注类别的顺序不一样，所以写了这个程序
"""
import os
from tqdm import tqdm
import random
import shutil

DEBUG_MODE = False  # print debug info if turn on


def change_class_order_in_labels(root_dir, t_head=0, t_body=1, ct_head=2, ct_body=3):
    """
    Change the original class order in image label txt files(yolo format with one *.txt file per image.) to following:
    0:t_head
    1:t_body
    2:ct_head
    3:ct_body

    Parameters:
        root_dir: root path of the dataset with 2 sub folder named "images" and "labels"
        t_head: int, original class index for t_head.
        ...
    Returns:
        None

    """
    # loop all *.txt under sub folder "labels"
    for filename in tqdm(os.listdir(os.path.join(root_dir, 'labels'))):

        lines = []
        output_lines = []

        if filename.endswith('.txt') and filename != "classes.txt":  # skip "classes.txt"
            # read mode
            with open(os.path.join(root_dir, 'labels', filename), 'r') as f:
                lines = f.readlines()

            # debug purpose
            if DEBUG_MODE:
                print('Before:')
                for line in lines:
                    print(line)

            # replace target class order with original class order.
            for line in lines:
                row_list = line.split(' ')  # Each row is class x_center y_center width height format.
                if row_list[0] == str(t_head):
                    row_list[0] = 0
                elif row_list[0] == str(t_body):
                    row_list[0] = 1
                elif row_list[0] == str(ct_head):
                    row_list[0] = 2
                else:  # ct_body
                    row_list[0] = 3
                changed_line = f'{row_list[0]} {row_list[1]} {row_list[2]} {row_list[3]} {row_list[4]}'
                output_lines.append(changed_line)

            # write mode(override, not append)
            with open(os.path.join(root_dir, 'labels', filename), 'w') as f:
                f.writelines(output_lines)

            # debug purpose. check if class was changed.
            if DEBUG_MODE:
                print('after:')
                with open(os.path.join(root_dir, 'labels', filename), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        print(line)

    # end of loop

    # update classes.txt under labels folder
    with open(os.path.join(root_dir, 'labels', 'classes.txt'), 'w') as f:
        f.writelines(['t_head\n', 't_body\n', 'ct_head\n', 'ct_body\n'])


def get_dataset_info(root_dir, sub_dir="train"):
    """
    Count the class distribution of the dataset.
    It seems yolov5 will do the same thing before training.
    Parameters:
        root_dir: root path of the dataset with 2 sub folder named "images" and "labels"
        sub_dir: "train", "val" or "test". Check the given dataset.
    Returns:
        None
    """
    # count the image and instance number in each class.
    class_counter = {
        "total_img_cnt": 0,
        "t_head_image_cnt": 0,
        "t_body_image_cnt": 0,
        "ct_head_image_cnt": 0,
        "ct_body_image_cnt": 0,
        "t_head_instance_cnt": 0,
        "t_body_instance_cnt": 0,
        "ct_head_instance_cnt": 0,
        "ct_body_instance_cnt": 0,
        "background_cnt": 0,  # Background images are images with no objects that are added to a dataset to reduce False
        # Positives (FP). No labels are required for background images.
        "background_set": set()
    }

    # loop all *.txt under sub folder "labels"
    for filename in tqdm(os.listdir(os.path.join(root_dir, 'labels', sub_dir))):

        if filename.endswith('.txt') and filename != "classes.txt":  # skip "classes.txt"
            # declare variables used to count the image number in each class.
            t_head_exist = False  # set to False at the beginning of each file.
            t_body_exist = False
            ct_head_exist = False
            ct_body_exist = False
            # read mode
            with open(os.path.join(root_dir, 'labels', sub_dir, filename), 'r') as f:
                lines = f.readlines()

            # count the instance number in each class.
            for line in lines:
                row_list = line.split(' ')  # Each row is class x_center y_center width height format.
                if row_list[0] == '0':
                    class_counter['t_head_instance_cnt'] += 1
                    t_head_exist = True
                elif row_list[0] == '1':
                    class_counter['t_body_instance_cnt'] += 1
                    t_body_exist = True
                elif row_list[0] == '2':
                    class_counter['ct_head_instance_cnt'] += 1
                    ct_head_exist = True
                else:  # ct_body
                    class_counter['ct_body_instance_cnt'] += 1
                    ct_body_exist = True

            # count the image number in each class.
            if t_head_exist:
                class_counter['t_head_image_cnt'] += 1
            if t_body_exist:
                class_counter['t_body_image_cnt'] += 1
            if ct_head_exist:
                class_counter['ct_head_image_cnt'] += 1
            if ct_body_exist:
                class_counter['ct_body_image_cnt'] += 1

    # end of loop

    # get background images info
    img_file_list = os.listdir(os.path.join(root_dir, 'images', sub_dir))
    # TBD: assume image is saved as "jpg" type
    # remove '.jpg' suffix and change list to set
    img_set = set([name.replace('.jpg', '').replace('.png', '')
                   for name in img_file_list if (name.endswith('.jpg') or name.endswith('.png'))])
    label_file_list = os.listdir(os.path.join(root_dir, 'labels', sub_dir))

    # 注意：发现部分别人数据集里的图片有对应的标签txt文件，但实际上为空文件，这里加入判断逻辑。
    # os.path.getsize() 返回文件的字节数，如果为 0，则代表空。当作没有这个文件，从列表中剔除！
    label_file_list = [f for f in label_file_list
                       if os.path.getsize(os.path.join(root_dir, 'labels', sub_dir, f))]

    # remove '.txt' suffix and so on
    label_set = set([name.replace('.txt', '') for name in label_file_list if name.endswith('.txt')])
    background_set = img_set-label_set  # Set_A - Set_B = {element in A but not in B}
    class_counter['total_img_cnt'] = len(img_set)
    class_counter['background_cnt'] = len(background_set)
    class_counter['background_set'] = background_set

    # print info
    print(f"{sub_dir} dataset info:")
    print(class_counter)
    print(f"Background images pct is {class_counter['background_cnt']/class_counter['total_img_cnt']*100:.2f}%")


def train_val_dataset_split(from_dir, output_dir, train_ratio=0.8):
    """
    Split the dataset to train and val dataset. Files will be copied.
    Original input dataset folder structure should be organized like the following:
    1. from_dir/images
    2. from_dir/labels
    Output folder structure will be:
    1. output_dir/images/train
    2. output_dir/images/val
    3. output_dir/labels/train
    4. output_dir/labels/val
    Parameters:
        from_dir: root path of the original dataset
        output_dir: output path.
        train_ratio: percentage of the training set.
    Returns:
        None
    """

    org_imgs_path = os.path.join(from_dir, 'images')
    org_labels_path = os.path.join(from_dir, 'labels')
    output_imgs_train_path = os.path.join(output_dir, 'images', 'train')
    output_imgs_val_path = os.path.join(output_dir, 'images', 'val')
    output_labels_train_path = os.path.join(output_dir, 'labels', 'train')
    output_labels_val_path = os.path.join(output_dir, 'labels', 'val')

    # create folders
    if not os.path.exists(output_imgs_train_path):
        os.makedirs(output_imgs_train_path)
    if not os.path.exists(output_imgs_val_path):
        os.makedirs(output_imgs_val_path)
    if not os.path.exists(output_labels_train_path):
        os.makedirs(output_labels_train_path)
    if not os.path.exists(output_labels_val_path):
        os.makedirs(output_labels_val_path)

    # random samples
    image_file_list = os.listdir(org_imgs_path)  # 取图片的原始路径
    image_cnt = len(image_file_list)
    train_number = int(image_cnt * train_ratio)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(image_file_list, train_number)  # 随机选取指定数量的样本图片

    for name in tqdm(image_file_list):
        label_name = name.split(".")[0] + '.txt'  # 注意：数据集中标注文件可能不存在
        if name in sample:  # if train
            # copy train images
            shutil.copy(os.path.join(org_imgs_path, name),
                        os.path.join(output_imgs_train_path, name))
            # copy train labels
            if os.path.exists(os.path.join(org_labels_path, label_name)):
                shutil.copy(os.path.join(org_labels_path, label_name),
                            os.path.join(output_labels_train_path, label_name))
        else:  # if val
            # copy val images
            shutil.copy(os.path.join(org_imgs_path, name),
                        os.path.join(output_imgs_val_path, name))
            # copy val labels
            if os.path.exists(os.path.join(org_labels_path, label_name)):
                shutil.copy(os.path.join(org_labels_path, label_name),
                            os.path.join(output_labels_val_path, label_name))


"""
# *********************************************************************************************************
# 查看原数据集，确认原始的标注顺序。注意文件夹结构，原始数据集下要有"images" and "labels"子文件夹，且labels文件夹下有classes.txt
# 启动labelimg的方法：
(建议！)使用参数运行labelimg，如下格式(参考资料：https://github.com/heartexlabs/labelImg)：
labelimg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
注意：路径不能包含中文，否则无法正确识别！可以包含空格，但变量就必须用引号，比如 
(启动后，可能需要change save dir才能正确将已标注的图片信息显示出来。)
其次但最重要的是，特别注意要指定预定义类别的文件比如叫predefined_classes.txt(可以任意指定路径，不包含中文即可)。如果没有指定的话，
第二次重新打开软件进行标注时，每次执行保存操作后，如果当前图片只有1类物体，则程序会自动更新labels文件夹下的classes.txt，变成1类。。
如果之后打开之前已经标注的图片，假如时已标注过3类物体，则会报错，list index out of range。正如下面两个github上的问题一样！！
https://github.com/heartexlabs/labelImg/issues/482
https://github.com/heartexlabs/labelImg/issues/738
如果启动时指定了[PRE-DEFINED CLASS FILE]，则每次保存图片时，会根据当前图片的类别和predefined_classes.txt共同更新labels文件夹下
的classes.txt，就不会出现class减少的issue了。
# *********************************************************************************************************
"""

# Pre-process the CSGO dataset

# 下面代码只是跑一次就行了，不要重复跑！
# ***************************************原始数据集已经处理完*****************************************
# root_dir = r'J:\Project\yolov5\mydatasets\csgo_600'  # root path of the dataset with 2 sub folder named "images" and "labels"
# # 1.run below if needed，注意检查标签顺序！位置顺序依次为t_head, t_body, ct_head, ct_body
# change_class_order_in_labels(root_dir, 3, 2, 1, 0)

# root_dir = r'J:\Project\yolov5\mydatasets\群主的 颠倒配置过的数据集_可用\q_label'
# change_class_order_in_labels(root_dir, 3, 2, 1, 0)
# root_dir = r'J:\Project\yolov5\mydatasets\csgo_400b\D1'
# change_class_order_in_labels(root_dir, 2, 3, 0, 1)
# root_dir = r'J:\Project\yolov5\mydatasets\csgo_400b\D3'
# change_class_order_in_labels(root_dir, 2, 3, 0, 1)
# root_dir = r'J:\Project\yolov5\mydatasets\csgo_400b\D4'
# change_class_order_in_labels(root_dir, 2, 3, 0, 1)
# root_dir = r'J:\Project\yolov5\mydatasets\csgo_225'
# change_class_order_in_labels(root_dir, 0, 2, 1, 3)
# root_dir = r'J:\Project\yolov5\mydatasets\csgo_6000'
# change_class_order_in_labels(root_dir, 1, 0, 3, 2)
# ***************************************原始数据集已经处理完*****************************************


# ************************************随机拆分数据集***************************************
# from_dir = r'J:\Project\yolov5\mydatasets\csgo_all_20221204'
# output_dir = r'J:\Project\yolov5\mydatasets\csgo_for_training4'
# train_val_dataset_split(from_dir, output_dir, 0.8)

# **********************************查看训练集类别的统计信息*********************************
root_dir = r'J:\Project\yolov5\mydatasets\csgo_for_training4'
# get class distribution of the dataset
get_dataset_info(root_dir, "train")
get_dataset_info(root_dir, "val")
