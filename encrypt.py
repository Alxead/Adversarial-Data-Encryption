ATTACK_EPS = 0.5
ATTACK_STEPSIZE = 0.1
ATTACK_STEPS = 100
NUM_WORKERS = 8
BATCH_SIZE = 200

import os
import time
import torch as ch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model
import numpy as np
from robustness.tools.vis_tools import show_image_row
from robustness.tools.label_maps import CLASS_DICT
from encrypt import save_image
from use_robustness import AlexNet


def crop_4(images, bs):

    for k in range(bs):
        images[k] = images[k].cpu().numpy()
        image0 = images[k][0, :, :, :]
        image1 = images[k][1, :, :, :]
        image2 = images[k][2, :, :, :]
        image3 = images[k][3, :, :, :]
        image0 = (image0.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        image1 = (image1.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        image2 = (image2.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        image3 = (image3.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        image0 = Image.fromarray(image0)
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)
        image3 = Image.fromarray(image3)

        patch0 = image0.crop((0, 0, 16, 16))
        patch1 = image1.crop((16, 0, 32, 16))
        patch2 = image2.crop((0, 16, 16, 32))
        patch3 = image3.crop((16, 16, 32, 32))
        # image0.save('/home/huyingdong/GAN_examples/tmp/image.png')
        # patch0.save('/home/huyingdong/GAN_examples/tmp/patch0.png')
        # patch1.save('/home/huyingdong/GAN_examples/tmp/patch1.png')
        # patch2.save('/home/huyingdong/GAN_examples/tmp/patch2.png')
        # patch3.save('/home/huyingdong/GAN_examples/tmp/patch3.png')

        target = Image.new('RGB', (32, 32))
        target.paste(patch0, (0, 0, 16, 16))
        target.paste(patch1, (16, 0, 32, 16))
        target.paste(patch2, (0, 16, 16, 32))
        target.paste(patch3, (16, 16, 32, 32))
        # target.save('/home/huyingdong/GAN_examples/tmp/pingjie.png')

        transform = transforms.Compose([transforms.ToTensor()])
        target = transform(target)
        images[k] = target

    return images


def dig_hole(im_adv, orig_im):

    batch_size = len(im_adv)
    modified_im_adv = []

    for i in range(batch_size):
        image0 = im_adv[i].cpu().numpy()
        image1 = orig_im[i].cpu().numpy()
        image0 = (image0.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        image1 = (image1.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        image0 = Image.fromarray(image0)
        image1 = Image.fromarray(image1)

        patch = image1.crop((4, 5, 14, 15))
        # patch.save('/home/huyingdong/GAN_examples/tmp/patch.png')
        image0.paste(patch, (4, 5, 14, 15))
        # image0.save('/home/huyingdong/GAN_examples/tmp/wadong.png')

        transform = transforms.Compose([transforms.ToTensor()])
        image0 = transform(image0)
        image0 = ch.unsqueeze(image0, 0)

        modified_im_adv.append(image0)

    modified_im_adv = ch.cat(modified_im_adv)

    return modified_im_adv


def dig_hole2(im_adv, orig_im):

    batch_size = len(im_adv)
    modified_im_adv = []

    for i in range(batch_size):
        image0 = im_adv[i].cpu().numpy()
        image1 = orig_im[i].cpu().numpy()
        image0 = (image0.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        image1 = (image1.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        image0 = Image.fromarray(image0)
        image1 = Image.fromarray(image1)

        patch = image1.crop((20, 20, 30, 30))
        # patch.save('/home/huyingdong/GAN_examples/tmp/patch.png')
        image0.paste(patch, (20, 20, 30, 30))
        # image0.save('/home/huyingdong/GAN_examples/tmp/wadong.png')

        transform = transforms.Compose([transforms.ToTensor()])
        image0 = transform(image0)
        image0 = ch.unsqueeze(image0, 0)

        modified_im_adv.append(image0)

    modified_im_adv = ch.cat(modified_im_adv)

    return modified_im_adv


def map_label_target(label):

    batch_size = len(label)
    targ_list = []

    for i in range(batch_size):
        if label[i] == 0:
            targ_list.append(8)
        elif label[i] == 1:
            targ_list.append(3)
        elif label[i] == 2:
            targ_list.append(1)
        elif label[i] == 3:
            targ_list.append(0)
        elif label[i] == 4:
            targ_list.append(2)
        elif label[i] == 5:
            targ_list.append(4)
        elif label[i] == 6:
            targ_list.append(9)
        elif label[i] == 7:
            targ_list.append(6)
        elif label[i] == 8:
            targ_list.append(7)
        elif label[i] == 9:
            targ_list.append(5)

    targ = ch.tensor(targ_list)

    return targ


def map_label_target2(label):
    batch_size = len(label)
    targ_list = []

    for i in range(batch_size):
        if label[i] == 0:
            targ_list.append(4)
        elif label[i] == 1:
            targ_list.append(2)
        elif label[i] == 2:
            targ_list.append(3)
        elif label[i] == 3:
            targ_list.append(5)
        elif label[i] == 4:
            targ_list.append(7)
        elif label[i] == 5:
            targ_list.append(1)
        elif label[i] == 6:
            targ_list.append(8)
        elif label[i] == 7:
            targ_list.append(0)
        elif label[i] == 8:
            targ_list.append(6)
        elif label[i] == 9:
            targ_list.append(9)

    targ = ch.tensor(targ_list)

    return targ


def map_label_target3(label):

    batch_size = len(label)
    targ_list = []

    for i in range(batch_size):
        if label[i] == 0:
            targ_list.append(6)
        elif label[i] == 1:
            targ_list.append(2)
        elif label[i] == 2:
            targ_list.append(9)
        elif label[i] == 3:
            targ_list.append(1)
        elif label[i] == 4:
            targ_list.append(0)
        elif label[i] == 5:
            targ_list.append(7)
        elif label[i] == 6:
            targ_list.append(5)
        elif label[i] == 7:
            targ_list.append(3)
        elif label[i] == 8:
            targ_list.append(4)
        elif label[i] == 9:
            targ_list.append(8)

    targ = ch.tensor(targ_list)

    return targ


def map_label_target4(label):

    batch_size = len(label)
    targ_list = []

    for i in range(batch_size):
        if label[i] == 0:
            targ_list.append(3)
        elif label[i] == 1:
            targ_list.append(9)
        elif label[i] == 2:
            targ_list.append(6)
        elif label[i] == 3:
            targ_list.append(1)
        elif label[i] == 4:
            targ_list.append(5)
        elif label[i] == 5:
            targ_list.append(8)
        elif label[i] == 6:
            targ_list.append(4)
        elif label[i] == 7:
            targ_list.append(7)
        elif label[i] == 8:
            targ_list.append(0)
        elif label[i] == 9:
            targ_list.append(2)

    targ = ch.tensor(targ_list)

    return targ


def map_label_target_attacker(label):

    batch_size = len(label)
    targ_list = []

    for i in range(batch_size):
        if label[i] == 0:
            targ_list.append(6)
        elif label[i] == 1:
            targ_list.append(0)
        elif label[i] == 2:
            targ_list.append(8)
        elif label[i] == 3:
            targ_list.append(2)
        elif label[i] == 4:
            targ_list.append(5)
        elif label[i] == 5:
            targ_list.append(9)
        elif label[i] == 6:
            targ_list.append(7)
        elif label[i] == 7:
            targ_list.append(3)
        elif label[i] == 8:
            targ_list.append(1)
        elif label[i] == 9:
            targ_list.append(4)

    targ = ch.tensor(targ_list)

    return targ



def main():
    ch.manual_seed(23)
    # device = ch.device("cuda:0")

    ds = CIFAR('/home/huyingdong/data/cifar10')

    # model = AlexNet()
    # /home/huyingdong/robust_lib_exp/logs/checkpoints/dir/0ea2b5f2-8de6-4535-9e80-ac12fd7e44fd/checkpoint.pt.best
    model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                                      resume_path='/home/huyingdong/GAN_examples/logs/checkpoints/dir/resnet50/checkpoint.pt.best')
    model.eval()

    # 不对训练集进行数据增强
    train_loader, test_loader = ds.make_loaders(workers=NUM_WORKERS,
                                                batch_size=BATCH_SIZE,
                                                data_aug=False,
                                                shuffle_train=False,
                                                )

    kwargs = {
        'constraint': '2',
        'eps': ATTACK_EPS,
        'step_size': ATTACK_STEPSIZE,
        'iterations': ATTACK_STEPS,
        'targeted': True,
        'do_tqdm': False
    }

    # generate_train_data(train_loader, model, kwargs)

    # generate_test_data(test_loader, model, kwargs)

    # generate_test_data_mixup(test_loader, model, kwargs)

    # generate_train_data_mixandcat(train_loader, model, kwargs)

    # generate_test_data_mixandcat(test_loader, model, kwargs)

    generate_advtest_data(test_loader, model, kwargs)


# 最简单的版本
def generate_train_data(train_loader, model, kwargs):
    train_image = []
    train_label = []
    for i, (im, label) in enumerate(train_loader, 0):
        time_ep = time.time()

        # 固定targ
        # targ = map_label_target(label)

        # 随机targ
        batch_size = len(label)
        targ = np.random.randint(10, size=batch_size)
        targ = ch.tensor(targ)


        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target==pred:", ch.sum(targ == ch.argmax(model_prediction, dim=1).cpu()))

        save_image('./image/im.png', im[0])
        save_image('./image/im_adv.png', im_adv[0])
        save_image('./image/im2.png', im[1])
        save_image('./image/im_adv2.png', im_adv[1])

        train_image.append(im_adv.cpu())
        train_label.append(label.cpu())
        print("process train batch %d" % (i))
        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(train_image, os.path.join('./data', 'advdataset', 'train_image_random'))
    ch.save(train_label, os.path.join('./data', 'advdataset', 'train_label_random'))
    print("save train set finished")


def generate_test_data(test_loader, model, kwargs):

    test_image = []
    test_label = []
    correct = 0
    for i, (im, label) in enumerate(test_loader, 0):
        time_ep = time.time()
        print("process testset batch %d" % (i))
        predict, _ = model(im)
        label_pred = ch.argmax(predict, dim=1)

        # 固定targ
        # targ = map_label_target(label_pred)
        # targ = map_label_target_attacker(label_pred)

        # 随机targ
        batch_size = len(label_pred)
        targ = np.random.randint(10, size=batch_size)
        targ = ch.tensor(targ)


        print("label == pred_label:", ch.sum(label == label_pred.cpu()))
        correct += ch.sum(label == label_pred.cpu()).item()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target==pred:", ch.sum(targ == ch.argmax(model_prediction.cpu(), dim=1)))

        save_image('./image/im.png', im[0])
        save_image('./image/im_adv.png', im_adv[0])
        save_image('./image/im2.png', im[1])
        save_image('./image/im_adv2.png', im_adv[1])

        test_image.append(im_adv.cpu())
        test_label.append(label.cpu())

        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(test_image, os.path.join('./data', 'advdataset', 'test_image_random'))
    ch.save(test_label, os.path.join('./data', 'advdataset', 'test_label_random'))
    print("save test set finished")
    print("orig test set acc:%.4f" % (correct / 10000))


# horiz_concat
def generate_train_data_horiz(train_loader, model, kwargs):
    train_image = []
    train_label = []
    for i, (im, label) in enumerate(train_loader, 0):
        time_ep = time.time()
        # targ = (label + 1) % 10
        targ = map_label_target(label)
        targ2 = map_label_target2(label)
        # print("label:", label)
        # print("target:", targ)
        # print("target2:", targ2)

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target==pred:", ch.sum(targ == ch.argmax(model_prediction, dim=1).cpu()))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target2==pred:", ch.sum(targ2 == ch.argmax(model_prediction2, dim=1).cpu()))

        # bs * 3 * height * width
        alpha = 0.5
        dim_W = int(alpha * im.size()[3])
        im_adv_left = im_adv[:, :, :, :dim_W]
        im_adv_right = im_adv2[:, :, :, dim_W:]
        im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)

        save_image('./image/im.png', im[0])
        save_image('./image/im_adv.png', im_adv[0])
        save_image('./image/im2.png', im[1])
        save_image('./image/im_adv2.png', im_adv[1])

        train_image.append(im_adv.cpu())
        train_label.append(label.cpu())
        print("process train batch %d" % (i))
        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(train_image, os.path.join('./data', 'advdataset', 'train_image_50_horiz'))
    ch.save(train_label, os.path.join('./data', 'advdataset', 'train_label_50_horiz'))
    print("save train set finished")


# test set先预测，再左右分别攻击，再concat
def generate_test_data_horiz(test_loader, model, kwargs):

    test_image = []
    test_label = []
    correct = 0
    for i, (im, label) in enumerate(test_loader, 0):
        time_ep = time.time()
        print("process testset batch %d" % (i))
        predict, _ = model(im)
        label_pred = ch.argmax(predict, dim=1)
        targ = map_label_target(label_pred)
        targ2 = map_label_target2(label_pred)

        print("label == pred_label:", ch.sum(label == label_pred.cpu()))
        correct += ch.sum(label == label_pred.cpu()).item()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target==pred:", ch.sum(targ == ch.argmax(model_prediction.cpu(), dim=1)))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target2==pred:", ch.sum(targ2 == ch.argmax(model_prediction2.cpu(), dim=1)))

        # bs * 3 * height * width
        alpha = 0.1
        dim_W = int(alpha * im.size()[3])
        im_adv_left = im_adv[:, :, :, :dim_W]
        im_adv_right = im_adv2[:, :, :, dim_W:]
        im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)

        save_image('./image/im.png', im[0])
        save_image('./image/im_adv.png', im_adv[0])
        save_image('./image/im2.png', im[1])
        save_image('./image/im_adv2.png', im_adv[1])

        test_image.append(im_adv.cpu())
        test_label.append(label.cpu())

        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(test_image, os.path.join('./data', 'advdataset', 'test_image_10_horiz'))
    ch.save(test_label, os.path.join('./data', 'advdataset', 'test_label_10_horiz'))
    print("save test set finished")
    print("orig test set acc:%.4f" % (correct / 10000))


# mixup
def generate_train_data_mixup(train_loader, model, kwargs):
    train_image = []
    train_label = []
    for i, (im, label) in enumerate(train_loader, 0):
        time_ep = time.time()
        # targ = (label + 1) % 10
        targ = map_label_target(label)
        targ2 = map_label_target2(label)
        # print("label:", label)
        # print("target:", targ)
        # print("target2:", targ2)

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target==pred:", ch.sum(targ == ch.argmax(model_prediction, dim=1).cpu()))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target2==pred:", ch.sum(targ2 == ch.argmax(model_prediction2, dim=1).cpu()))

        # mixup
        alpha = 0.5
        im_adv = alpha * im_adv + (1 - alpha) * im_adv2

        save_image('./image/im.png', im[0])
        save_image('./image/im_adv.png', im_adv[0])
        save_image('./image/im2.png', im[1])
        save_image('./image/im_adv2.png', im_adv[1])

        train_image.append(im_adv.cpu())
        train_label.append(label.cpu())
        print("process train batch %d" % (i))
        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(train_image, os.path.join('./data', 'advdataset', 'train_image_50_mixup'))
    ch.save(train_label, os.path.join('./data', 'advdataset', 'train_label_50_mixup'))
    print("save train set finished")


# mixup
def generate_test_data_mixup(test_loader, model, kwargs):

    test_image = []
    test_label = []
    correct = 0
    for i, (im, label) in enumerate(test_loader, 0):
        time_ep = time.time()
        print("process testset batch %d" % (i))
        predict, _ = model(im)
        label_pred = ch.argmax(predict, dim=1)
        targ = map_label_target(label_pred)
        targ2 = map_label_target2(label_pred)

        print("label == pred_label:", ch.sum(label == label_pred.cpu()))
        correct += ch.sum(label == label_pred.cpu()).item()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target==pred:", ch.sum(targ == ch.argmax(model_prediction.cpu(), dim=1)))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target2==pred:", ch.sum(targ2 == ch.argmax(model_prediction2.cpu(), dim=1)))

        # mixup
        alpha = 0.1
        im_adv = alpha * im_adv + (1 - alpha) * im_adv2

        save_image('./image/im.png', im[0])
        save_image('./image/im_adv.png', im_adv[0])
        save_image('./image/im2.png', im[1])
        save_image('./image/im_adv2.png', im_adv[1])

        test_image.append(im_adv.cpu())
        test_label.append(label.cpu())

        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(test_image, os.path.join('./data', 'advdataset', 'test_image_10_mixup'))
    ch.save(test_label, os.path.join('./data', 'advdataset', 'test_label_10_mixup'))
    print("save test set finished")
    print("orig test set acc:%.4f" % (correct / 10000))


# mixup and concat (一张图攻击成四个)
def generate_train_data_mixandcat(train_loader, model, kwargs):
    train_image = []
    train_label = []
    for i, (im, label) in enumerate(train_loader, 0):
        time_ep = time.time()
        # targ = (label + 1) % 10
        targ = map_label_target(label)
        targ2 = map_label_target2(label)
        targ3 = map_label_target3(label)
        targ4 = map_label_target4(label)
        # print("label:", label)
        # print("target:", targ)
        # print("target2:", targ2)

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target==pred:", ch.sum(targ == ch.argmax(model_prediction, dim=1).cpu()))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target2==pred:", ch.sum(targ2 == ch.argmax(model_prediction2, dim=1).cpu()))

        model_prediction3, im_adv3 = model(im, targ3, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target3==pred:", ch.sum(targ3 == ch.argmax(model_prediction3, dim=1).cpu()))

        model_prediction4, im_adv4 = model(im, targ4, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target4==pred:", ch.sum(targ4 == ch.argmax(model_prediction4, dim=1).cpu()))

        # mixup and horizontal concat
        # bs * 3 * height * width
        lambd = 0.5
        alpha = 0.5
        im_adv = lambd * im_adv + (1 - lambd) * im_adv2
        im_adv3 = lambd * im_adv3 + (1 - lambd) * im_adv4
        dim_W = int(alpha * im.size()[3])
        im_adv_left = im_adv[:, :, :, :dim_W]
        im_adv_right = im_adv3[:, :, :, dim_W:]
        im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)

        save_image('./image/im.png', im[0])
        save_image('./image/im_adv.png', im_adv[0])
        save_image('./image/im2.png', im[1])
        save_image('./image/im_adv2.png', im_adv[1])

        train_image.append(im_adv.cpu())
        train_label.append(label.cpu())
        print("process train batch %d" % (i))
        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(train_image, os.path.join('./data', 'advdataset', 'train_image_mixandcat'))
    ch.save(train_label, os.path.join('./data', 'advdataset', 'train_label_mixandcat'))
    print("save train set finished")


# mixup and concat (一张图攻击成四个)
def generate_test_data_mixandcat(test_loader, model, kwargs):

    test_image = []
    test_label = []
    correct = 0
    for i, (im, label) in enumerate(test_loader, 0):
        time_ep = time.time()
        print("process testset batch %d" % (i))
        predict, _ = model(im)
        label_pred = ch.argmax(predict, dim=1)
        targ = map_label_target(label_pred)
        targ2 = map_label_target2(label_pred)
        targ3 = map_label_target3(label_pred)
        targ4 = map_label_target4(label_pred)

        print("label == pred_label:", ch.sum(label == label_pred.cpu()))
        correct += ch.sum(label == label_pred.cpu()).item()

        model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target==pred:", ch.sum(targ == ch.argmax(model_prediction.cpu(), dim=1)))

        model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target2==pred:", ch.sum(targ2 == ch.argmax(model_prediction2.cpu(), dim=1)))

        model_prediction3, im_adv3 = model(im, targ3, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target3==pred:", ch.sum(targ3 == ch.argmax(model_prediction3.cpu(), dim=1)))

        model_prediction4, im_adv4 = model(im, targ4, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target4==pred:", ch.sum(targ4 == ch.argmax(model_prediction4.cpu(), dim=1)))

        # mixup and horizontal concat
        # bs * 3 * height * width
        lambd = 0.5
        alpha = 0.5
        im_adv = lambd * im_adv + (1 - lambd) * im_adv2
        im_adv3 = lambd * im_adv3 + (1 - lambd) * im_adv4
        dim_W = int(alpha * im.size()[3])
        im_adv_left = im_adv[:, :, :, :dim_W]
        im_adv_right = im_adv3[:, :, :, dim_W:]
        im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)

        save_image('./image/im.png', im[0])
        save_image('./image/im_adv.png', im_adv[0])
        save_image('./image/im2.png', im[1])
        save_image('./image/im_adv2.png', im_adv[1])

        test_image.append(im_adv.cpu())
        test_label.append(label.cpu())

        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(test_image, os.path.join('./data', 'advdataset', 'test_image_mixandcat'))
    ch.save(test_label, os.path.join('./data', 'advdataset', 'test_label_mixandcat'))
    print("save test set finished")
    print("orig test set acc:%.4f" % (correct / 10000))


# 只攻击label为0的图片
def generate_advtest_data(test_loader, model, kwargs):
    test_image = []
    test_label = []
    correct = 0
    for i, (im, label) in enumerate(test_loader, 0):
        time_ep = time.time()
        print("process testset batch %d" % (i))
        predict, _ = model(im)
        label_pred = ch.argmax(predict, dim=1)
        print("label == pred_label:", ch.sum(label == label_pred.cpu()))
        correct += ch.sum(label == label_pred.cpu()).item()

        index = (label_pred == 0)
        total = ch.sum(index)
        if total == 0:
            index[0] = 1
            total = 1
        print("Number of pictures labeled with 0: ", total.item())
        im_label0 = im[index]
        targ = ch.zeros(total, dtype=ch.int64) + 9
        # targ2 = ch.zeros(total, dtype=ch.int64) + 3

        model_prediction, im_adv = model(im_label0, targ, make_adv=True, **kwargs)
        # print("model_prediction", ch.argmax(model_prediction, dim=1))
        print("target==pred:", ch.sum(targ == ch.argmax(model_prediction.cpu(), dim=1)))

        # model_prediction2, im_adv2 = model(im_label0, targ2, make_adv=True, **kwargs)
        # # print("model_prediction", ch.argmax(model_prediction, dim=1))
        # print("target2==pred:", ch.sum(targ2 == ch.argmax(model_prediction2.cpu(), dim=1)))

        # bs * 3 * height * width
        # alpha = 0.5
        # dim_W = int(alpha * im.size()[3])
        # im_adv_left = im_adv[:, :, :, :dim_W]
        # im_adv_right = im_adv2[:, :, :, dim_W:]
        # im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)
        # im_adv = im_adv.cpu()

        im[index] = im_adv.cpu()
        test_image.append(im)
        test_label.append(label.cpu())

        time_ep = time.time() - time_ep
        print("time:%.4f" % (time_ep))

    ch.save(test_image, os.path.join('./data', 'advdataset', 'test_image_attack0_9'))
    ch.save(test_label, os.path.join('./data', 'advdataset', 'test_label_attack0_9'))
    print("save test set finished")
    print("orig test set acc:%.4f" % (correct / 10000))


########################################################
#测试集从0-9依次攻击
# test_image = []
# test_label = []
# for i, (im, label) in enumerate(test_loader, 0):
#     time_ep = time.time()
#     print("process testset batch %d" % (i))
#     print("label:", label)
#     batch_size = len(im)
#     batch_im = []
#
#     for t in range(10):
#         targ = ch.full_like(label, t)
#         print("target:", targ)
#         model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
#         print("model_pred:", ch.argmax(model_prediction, dim=1))
#
#         im_adv = im_adv.chunk(batch_size, 0)
#
#         for k in range(batch_size):
#             if t == 0:
#                 batch_im.append(im_adv[k])
#             else:
#                 batch_im[k] = ch.cat((batch_im[k], im_adv[k]))
#
#
#     for k in range(batch_size):
#         batch_im[k] = ch.unsqueeze(batch_im[k], 0)
#
#     batch_im = ch.cat(batch_im)
#
#     test_image.append(batch_im.cpu())
#     test_label.append(label.cpu())
#     time_ep = time.time() - time_ep
#     print("time:%.4f" % (time_ep))
#
#
# ch.save(test_image, os.path.join('./data', 'advdataset', '10_test_image'))
# ch.save(test_label, os.path.join('./data', 'advdataset', '10_test_label'))
# print("save test set finished")



#####################################################
# 剪成4份后合并
# test_image = []
# test_label = []
# for i, (im, label) in enumerate(test_loader, 0):
#     time_ep = time.time()
#     print("process testset batch %d" % (i))
#     print("label:", label)
#     batch_size = len(im)
#     batch_im = []
#
#     for t in range(4):
#         targ = ch.full_like(label, t)
#         print("target:", targ)
#         model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
#         print("model_pred:", ch.argmax(model_prediction, dim=1))
#
#         im_adv = im_adv.chunk(batch_size, 0)
#
#         for k in range(batch_size):
#             if t == 0:
#                 batch_im.append(im_adv[k])
#             else:
#                 batch_im[k] = ch.cat((batch_im[k], im_adv[k]))
#
#     batch_im = crop_4(batch_im, batch_size)
#
#     for k in range(batch_size):
#         batch_im[k] = ch.unsqueeze(batch_im[k], 0)
#
#     batch_im = ch.cat(batch_im)
#
#     test_image.append(batch_im.cpu())
#     test_label.append(label.cpu())
#     time_ep = time.time() - time_ep
#     print("time:%.4f" % (time_ep))
#
# ch.save(test_image, os.path.join('./data', 'advdataset', 'test_image'))
# ch.save(test_label, os.path.join('./data', 'advdataset', 'test_label'))
# print("save test set finished")
#
# train_image = []
# train_label = []
# for i, (im, label) in enumerate(train_loader, 0):
#     time_ep = time.time()
#     print("process trainset batch %d" % (i))
#     print("label:", label)
#     batch_size = len(im)
#     batch_im = []
#
#     for t in range(4):
#         targ = ch.full_like(label, t)
#         print("target:", targ)
#         model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
#         print("model_pred:", ch.argmax(model_prediction, dim=1))
#
#         im_adv = im_adv.chunk(batch_size, 0)
#
#         for k in range(batch_size):
#             if t == 0:
#                 batch_im.append(im_adv[k])
#             else:
#                 batch_im[k] = ch.cat((batch_im[k], im_adv[k]))
#
#     batch_im = crop_4(batch_im, batch_size)
#
#     for k in range(batch_size):
#         batch_im[k] = ch.unsqueeze(batch_im[k], 0)
#
#     batch_im = ch.cat(batch_im)
#
#     train_image.append(batch_im.cpu())
#     train_label.append(label.cpu())
#     time_ep = time.time() - time_ep
#     print("time:%.4f" % (time_ep))
#
# ch.save(train_image, os.path.join('./data', 'advdataset', 'train_image'))
# ch.save(train_label, os.path.join('./data', 'advdataset', 'train_label'))
# print("train test set finished")


########################################################################
## 向（y+1）mod C 攻击
# train_image = []
# train_label = []
# for i, (im, label) in enumerate(train_loader, 0):
#     time_ep = time.time()
#     # targ = (label + 1) % 10
#     targ = map_label_target(label)
#     print("label:", label)
#     print("target:", targ)
#
#     model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
#     print("model_prediction", ch.argmax(model_prediction, dim=1))
#     print("target==pred:", ch.sum(targ == ch.argmax(model_prediction, dim=1).cpu()))
#
#     # 加密图片挖一个洞
#     im_adv = dig_hole(im_adv, im)
#
#     train_image.append(im_adv.cpu())
#     train_label.append(label.cpu())
#     print("process train batch %d" % (i))
#     time_ep = time.time() - time_ep
#     print("time:%.4f" % (time_ep))
#
# ch.save(train_image, os.path.join('./data', 'advdataset', 'train_image'))
# ch.save(train_label, os.path.join('./data', 'advdataset', 'train_label'))
# print("save train set finished")

##########################################################
####### horiz_concat
# train_image = []
# train_label = []
# for i, (im, label) in enumerate(train_loader, 0):
#     time_ep = time.time()
#     # targ = (label + 1) % 10
#     targ = map_label_target(label)
#     targ2 = map_label_target2(label)
#     print("label:", label)
#     print("target:", targ)
#     print("target2:", targ2)
#
#     model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
#     # print("model_prediction", ch.argmax(model_prediction, dim=1))
#     print("target==pred:", ch.sum(targ == ch.argmax(model_prediction, dim=1).cpu()))
#
#     model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
#     # print("model_prediction", ch.argmax(model_prediction, dim=1))
#     print("target2==pred:", ch.sum(targ2 == ch.argmax(model_prediction2, dim=1).cpu()))
#
#     # bs * 3 * height * width
#     dim_W = 16
#     im_adv_left = im_adv[:, :, :, :dim_W]
#     im_adv_right = im_adv2[:, :, :, dim_W:]
#     im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)
#
#     train_image.append(im_adv.cpu())
#     train_label.append(label.cpu())
#     print("process train batch %d" % (i))
#     time_ep = time.time() - time_ep
#     print("time:%.4f" % (time_ep))
#
# ch.save(train_image, os.path.join('./data', 'advdataset', 'train_image_50_horiz'))
# ch.save(train_label, os.path.join('./data', 'advdataset', 'train_label_50_horiz'))
# print("save train set finished")


##########################################################
## test set 先预测再向（y+1）mod C 攻击
# test_image = []
# test_label = []
# correct = 0
# for i, (im, label) in enumerate(test_loader, 0):
#
#     time_ep = time.time()
#     print("process testset batch %d" % (i))
#     # batch_size = len(label)
#     # targ = np.random.randint(10, size=batch_size)
#     # targ = ch.tensor(targ)
#     predict, _ = model(im)
#     label_pred = ch.argmax(predict, dim=1)
#     targ = map_label_target(label_pred)
#
#     print("label:", label)
#     print("label_pred:", label_pred)
#     print("label == pred_label:", ch.sum(label == label_pred.cpu()))
#     correct += ch.sum(label == label_pred.cpu()).item()
#     print("target:", targ)
#
#     model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
#     print("model_prediction", ch.argmax(model_prediction, dim=1))
#     print("target==pred:", ch.sum(targ == ch.argmax(model_prediction.cpu(), dim=1)))
#
#     # 加密图片挖一个洞
#     im_adv = dig_hole2(im_adv, im)
#
#     # index = (label != 9)
#     # im_adv = im_adv[index]
#     # label = label[index]
#     save_image('./image/im.png', im[0])
#     save_image('./image/im_adv.png', im_adv[0])
#     save_image('./image/im2.png', im[1])
#     save_image('./image/im_adv2.png', im_adv[1])
#
#     test_image.append(im_adv.cpu())
#     test_label.append(label.cpu())
#
#     time_ep = time.time() - time_ep
#     print("time:%.4f" % (time_ep))
#
# ch.save(test_image, os.path.join('./data', 'advdataset', 'test_image_hole2'))
# ch.save(test_label, os.path.join('./data', 'advdataset', 'test_label_hole2'))
# print("save test set finished")
# print("orig test set acc:%.4f" % (correct / 10000))


##########################################################
#test set先预测，再左右分别攻击，再concat
# test_image = []
# test_label = []
# correct = 0
# for i, (im, label) in enumerate(test_loader, 0):
#
#     time_ep = time.time()
#     print("process testset batch %d" % (i))
#     predict, _ = model(im)
#     label_pred = ch.argmax(predict, dim=1)
#     targ = map_label_target(label_pred)
#     targ2 = map_label_target2(label_pred)
#
#     print("label:", label)
#     print("label_pred:", label_pred)
#     print("label == pred_label:", ch.sum(label == label_pred.cpu()))
#     correct += ch.sum(label == label_pred.cpu()).item()
#     print("target:", targ)
#
#     model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
#     # print("model_prediction", ch.argmax(model_prediction, dim=1))
#     print("target==pred:", ch.sum(targ == ch.argmax(model_prediction.cpu(), dim=1)))
#
#     model_prediction2, im_adv2 = model(im, targ2, make_adv=True, **kwargs)
#     # print("model_prediction", ch.argmax(model_prediction, dim=1))
#     print("target2==pred:", ch.sum(targ2 == ch.argmax(model_prediction2.cpu(), dim=1)))
#
#     # bs * 3 * height * width
#     alpha = 0.1
#     dim_W = int(alpha * im.size()[3])
#     im_adv_left = im_adv[:, :, :, :dim_W]
#     im_adv_right = im_adv2[:, :, :, dim_W:]
#     im_adv = ch.cat((im_adv_left, im_adv_right), dim=3)
#
#     # index = (label != 9)
#     # im_adv = im_adv[index]
#     # label = label[index]
#     save_image('./image/im.png', im[0])
#     save_image('./image/im_adv.png', im_adv[0])
#     save_image('./image/im2.png', im[1])
#     save_image('./image/im_adv2.png', im_adv[1])
#
#     test_image.append(im_adv.cpu())
#     test_label.append(label.cpu())
#
#     time_ep = time.time() - time_ep
#     print("time:%.4f" % (time_ep))
#
# ch.save(test_image, os.path.join('./data', 'advdataset', 'test_image_10_horiz'))
# ch.save(test_label, os.path.join('./data', 'advdataset', 'test_label_10_horiz'))
# print("save test set finished")
# print("orig test set acc:%.4f" % (correct / 10000))


# ##################################################################
# 攻击(y+1) % 10
# train_image = []
# train_label = []
# for i, (im, label) in enumerate(train_loader, 0):
#
#     # batch_size = len(label)
#     # targ = np.random.randint(10, size=batch_size)
#     # targ = ch.tensor(targ)
#
#     targ = (label+1) % 10
#     print("label:", label)
#     print("target:", targ)
#
#     model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
#     print("model_prediction", ch.argmax(model_prediction, dim=1))
#     print("target==pred:", ch.sum(targ == ch.argmax(model_prediction, dim=1).cpu()))
#
#     train_image.append(im_adv.cpu())
#     train_label.append(label.cpu())
#     print("process train batch %d" % (i))
#
# ch.save(train_image, os.path.join('./data', 'advdataset', 'train_image'))
# ch.save(train_label, os.path.join('./data', 'advdataset', 'train_label'))
# print("save train set finished")

##################################################################
# 存原始数据和加密数据,加密数据和原始数据cat起来
# train_image = []
# train_label = []
# cat_train_image = []
# cat_train_label = []
# for i, (im, label) in enumerate(train_loader, 0):
#     time_ep = time.time()
#     print("process train batch %d" % (i))
#     # batch_size = len(label)
#     # targ = np.random.randint(10, size=batch_size)
#     # targ = ch.tensor(targ)
#
#     targ = (label+1) % 10
#     print("label:", label)
#     print("target:", targ)
#
#     model_prediction, im_adv = model(im, targ, make_adv=True, **kwargs)
#     print("model_prediction", ch.argmax(model_prediction, dim=1))
#     print("target==pred:", ch.sum(targ == ch.argmax(model_prediction, dim=1).cpu()))
#
#     save_image('./image/im_adv.png', im_adv[0])
#     save_image('./image/im2.png', im[1])
#     save_image('./image/im_adv2.png', im_adv[1])
#
#     cat_im = ch.cat((im, im_adv.cpu()), dim=1)
#
#     train_image.append(im_adv.cpu())
#     train_label.append(label.cpu())
#     cat_train_image.append(cat_im.cpu())
#     cat_train_label.append(label.cpu())
#
#     time_ep = time.time() - time_ep
#     print("time:%.4f" % (time_ep))
#
#
# ch.save(train_image, os.path.join('./data', 'advdataset', 'train_image'))
# ch.save(train_label, os.path.join('./data', 'advdataset', 'train_label'))
# ch.save(cat_train_image, os.path.join('./data', 'advdataset', 'cat_train_image'))
# ch.save(cat_train_label, os.path.join('./data', 'advdataset', 'cat_train_label'))
# print("save train set finished")
#########################################################


# _, (im, label) = next(enumerate(test_loader))
# print(label)
# targ = ch.full_like(label, 2)
#
# tmp, im_adv = model(im, targ, make_adv=True, **kwargs)
# print(ch.argmax(tmp, dim=1))
#
# # Get predicted labels for adversarial examples
# pred, _ = model(im_adv)
# label_pred = ch.argmax(pred, dim=1)
# print(label_pred)
#
#
# # Visualize test set images, along with corresponding adversarial examples
# show_image_row([im.cpu(), im_adv.cpu()],
#          tlist=[[CLASS_DICT['CIFAR'][int(t)] for t in l] for l in [label, label_pred]],
#          fontsize=18,
#          filename='./adversarial_example_CIFAR.png')

if __name__ == '__main__':
    main()

