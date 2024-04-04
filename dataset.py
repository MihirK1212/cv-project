import os
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

import config
import utils

class_id_to_class_label = dict()

def read_img(img_path):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image or numpy.ndarray to tensor
    ])
    image = Image.open(img_path)
    image_tensor = transform(image)
    assert image_tensor.shape == (3, config.IMG_SIZE, config.IMG_SIZE)
    return image_tensor

def get_train_dataset():

    X_train, y_train = [], []

    TRAIN_BASE_PATH = './tiny-imagenet-200/tiny-imagenet-200/train'

    global class_id_to_class_label
    curr_class_label = 0

    count_train_error = 0

    for class_id in os.listdir(TRAIN_BASE_PATH):
        assert str(class_id) not in class_id_to_class_label
        class_id_to_class_label[str(class_id)] = curr_class_label
        curr_class_label+=1

        for img_file_name in os.listdir(os.path.join(TRAIN_BASE_PATH, class_id, 'images')):
            try:
                img_path = os.path.join(TRAIN_BASE_PATH, class_id, 'images', img_file_name)
                img = read_img(img_path)
                X_train.append(img)
                y_train.append(class_id_to_class_label[str(class_id)])
            except Exception as E:
                # print('error in reading train image', img_file_name, E)
                count_train_error+=1

        print(f'train {class_id} done, {len(X_train)} images, {X_train[0].shape} shape')

    print('number of train errors', count_train_error)

    return X_train, y_train

def get_valid_dataset():

    X_valid, y_valid = [], []

    VALID_BASE_PATH = 'tiny-imagenet-200/tiny-imagenet-200/val/images'
    VALID_ANNOTATIONS_PATH = 'tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt'

    global class_id_to_class_label

    count_valid_error = 0

    img_file_name_to_class_id = dict()
    with open(VALID_ANNOTATIONS_PATH, 'r') as f:
        for line in f:
            try:
                spl_line = line.split('\t')
                img_file_name, class_id = spl_line[0], spl_line[1]
                img_path = os.path.join(VALID_BASE_PATH, img_file_name)
                img = read_img(img_path)
                class_label = class_id_to_class_label[str(class_id)]
                X_valid.append(img)
                y_valid.append(class_label)
            except Exception as E:
                # print('error in reading valid image', img_file_name, E)
                count_valid_error+=1

    print('number of valid errors', count_valid_error)

    return X_valid, y_valid
    

def get_test_dataset(num_samples = config.NUM_RANDOM_SAMPLES):
  X_test = [torch.rand(3, config.IMG_SIZE, config.IMG_SIZE).to(utils.get_device()).cuda() for _ in range(num_samples)]
  y_test = [torch.randint(0, config.NUM_CLASSES, (1,)).item() for _ in range(num_samples)]
  return X_test, y_test

def get_dataset():
    if config.USE_RANDOM_DATASET:
        X_train, y_train = get_test_dataset()
        X_valid, y_valid = get_test_dataset()
        X_test, y_test = get_test_dataset()
    else:
        X_train, y_train = get_train_dataset()
        X_valid, y_valid = get_valid_dataset()
        X_test, y_test = get_test_dataset()
    return X_train, X_valid, X_test, y_train, y_valid, y_test
   
# def get_dataset():
#     X, y = get_test_dataset()
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)