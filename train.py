# -*-coding:utf-8-*-
import argparse
import logging
import yaml
import time
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict
from models import *

from utils import Logger, count_parameters, data_augmentation, \
    load_checkpoint, get_data_loader, mixup_data, mixup_criterion, \
    save_checkpoint, adjust_learning_rate, get_current_lr

parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()
logger = Logger(log_file_name=args.work_path + '/log.txt',
                log_level=logging.DEBUG, logger_name="CIFAR").get_log()


def predicted_oneclass(label, predicted, one_class=0):

    result = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}

    index = (label == one_class)
    number_of_oneclass = torch.sum(index).item()
    selected = predicted[index].numpy()

    for i in range(number_of_oneclass):
        if selected[i] == 0:
            result['0'] += 1
        if selected[i] == 1:
            result['1'] += 1
        if selected[i] == 2:
            result['2'] += 1
        if selected[i] == 3:
            result['3'] += 1
        if selected[i] == 4:
            result['4'] += 1
        if selected[i] == 5:
            result['5'] += 1
        if selected[i] == 6:
            result['6'] += 1
        if selected[i] == 7:
            result['7'] += 1
        if selected[i] == 8:
            result['8'] += 1
        if selected[i] == 9:
            result['9'] += 1

    return result


def merge_dict(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] += value
        else:
            dict1[key] += value
    return dict1


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

    targ = torch.tensor(targ_list)

    return targ


def train(train_loader, net, criterion, optimizer, epoch, device):
    global writer

    start = time.time()
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, config.mixup_alpha, device)

            outputs = net(inputs)
            loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        # zero the gradient buffers
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weight
        optimizer.step()

        # count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.mixup:
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

        if (batch_index + 1) % 100 == 0:
            logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader),
                train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader),
        train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total

    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)

    return train_loss, train_acc

#########################
# v1 test (Not modified)
#########################

def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec, writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" *** Validate Encrypt ***".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)
    # Save checkpoint.
    acc = 100. * correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
    if is_best:
        best_prec = acc

# def test(test_loader, net, criterion, optimizer, epoch, device):
#     global best_prec, writer
#
#     net.eval()
#
#     test_loss = 0
#     correct = 0
#     total = 0
#
#     logger.info(" *** Validate Encrypt ***".format(epoch + 1, config.epochs))
#
#     with torch.no_grad():
#         for batch_index, (inputs, targets) in enumerate(test_loader):
#
#             confidence = torch.full_like(targets, -100.0).float().to(device)
#             predicted = torch.zeros_like(targets).to(device)
#             for i in range(10):
#                 images = inputs[:, i, ...]
#                 images, targets = images.to(device), targets.to(device)
#                 outputs = net(images)
#                 # soft_outputs = nn.functional.softmax(outputs, dim=1)
#                 c_confidence, c_predicted = outputs.max(1)
#                 predicted[c_confidence > confidence] = c_predicted[c_confidence > confidence]
#                 confidence[c_confidence > confidence] = c_confidence[c_confidence > confidence]
#
#             loss = criterion(outputs, targets)
#
#             test_loss += loss.item()
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#     logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
#         test_loss / (batch_index + 1), 100.0 * correct / total))
#     test_loss = test_loss / (batch_index + 1)
#     test_acc = correct / total
#     writer.add_scalar('test_loss', test_loss, global_step=epoch)
#     writer.add_scalar('test_acc', test_acc, global_step=epoch)
#     # Save checkpoint.
#     acc = 100. * correct / total
#     state = {
#         'state_dict': net.state_dict(),
#         'best_prec': best_prec,
#         'last_epoch': epoch,
#         'optimizer': optimizer.state_dict(),
#     }
#     is_best = acc > best_prec
#     save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
#     if is_best:
#         best_prec = acc

def test_orig(test_loader, net, criterion, epoch, device):
    global writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" *** Validate Orig ***".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)


def test_orig_detail(test_loader, net, criterion, epoch, device):
    global writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    ten_dict = {'label0': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0},
                'label1': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0},
                'label2': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0},
                'label3': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0},
                'label4': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0},
                'label5': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0},
                'label6': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0},
                'label7': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0},
                'label8': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0},
                'label9': {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
                }

    logger.info(" *** Validate Orig ***".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            ####
            # result_oneclass = predicted_oneclass(targets.cpu(), predicted.cpu(), one_class=8)
            # for key, value in result_oneclass.items():
            #     if key in ten_dict:
            #         ten_dict[key] += value
            #     else:
            #         ten_dict[key] += value

            for i in range(10):
                result_oneclass = predicted_oneclass(targets.cpu(), predicted.cpu(), one_class=i)
                k = 'label' + str(i)
                ten_dict[k] = merge_dict(ten_dict[k], result_oneclass)

            ####


            ## 修正后的acc 比真实的高
            predicted = map_label_target(predicted)
            predicted = predicted.to(device)
            ##

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)

    for labe, one_dict in ten_dict.items():
        print("true label:", labe)
        for key, value in one_dict.items():
            print(key+':'+str(100.0 * value / 1000)+'%')
        print('--------------------')



def main():
    global args, config, last_epoch, best_prec, writer
    writer = SummaryWriter(log_dir=args.work_path + '/event')

    # read config from yaml file
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)

    # define netowrk
    net = get_model(config)
    logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = 'cuda:2' if config.use_gpu else 'cpu'
    # data parallel for multiple-GPU
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        config.lr_scheduler.base_lr,
        momentum=config.optimize.momentum,
        weight_decay=config.optimize.weight_decay,
        nesterov=config.optimize.nesterov)

    # resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    if args.work_path:
        ckpt_file_name = args.work_path + '/' + config.ckpt_name + '.pth.tar'
        if args.resume:
            best_prec, last_epoch = load_checkpoint(
                ckpt_file_name, net, optimizer=optimizer)

    # load training data, do data augmentation and get data loader
    transform_train = transforms.Compose(
        data_augmentation(config))

    transform_test = transforms.Compose(
        data_augmentation(config, is_train=False))

    loaders = get_data_loader(
        transform_train, transform_test, config)

    train_loader = loaders['enc_train']
    test_loader = loaders['enc_test']
    test_loader_orig = loaders['orig_test']

    logger.info("            =======  Training  =======\n")
    for epoch in range(last_epoch + 1, config.epochs):
        lr = adjust_learning_rate(optimizer, epoch, config)
        writer.add_scalar('learning_rate', lr, epoch)
        train(train_loader, net, criterion, optimizer, epoch, device)
        if epoch == 0 or (
                epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)
            test_orig(test_loader_orig, net, criterion, epoch, device)
    writer.close()
    logger.info(
        "======== Training Finished.   best_test_acc: {:.3f}% ========".format(best_prec))


if __name__ == "__main__":
    main()
