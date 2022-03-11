import os
import math
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from my_dataset import MyDataSet
from train_one_epoch import train_one_epoch
from evaluate import evaluate
from utils import *

# 导入模型
from models.resnet import resnet50
from models.densenet import densenet121
from models.shufflenet import shufflenet_v2_x1_0


def train(opt, last_best_acc=None):
    device = select_device(opt.device)
    epochs = opt.epochs
    weights = opt.weights
    batch_size = opt.batch_size
    lr = opt.lr
    lrf = opt.lrf
    momentum = opt.momentum
    eps = opt.eps
    weight_decay = opt.weight_decay
    freeze_layers = opt.freeze_layers
    use_adam = opt.adam
    workers = opt.workers
    image_path = opt.data_path
    save_path = opt.save_path
    evolve = opt.evolve

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # 打开 tensorboard
    tb_writer = SummaryWriter()
    logger.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    # 加载数据
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(image_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # 多线程
    nw = min([workers, os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    logger.info('Using {} dataloader workers every process.'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 加载模型结构
    model = resnet50(num_classes=5).to(device)

    # 加载模型权重，如果有的话
    if weights != "":
        if os.path.exists(weights):
            weights_dict = torch.load(weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            logger.info(model.load_state_dict(load_weights_dict, strict=False))
        else:
            logger.error("ERROR! weights doesn't exist.")
            sys.exit()

    # # 查看参数名称，以便选择需要训练的参数
    # for name, para in model.named_parameters():
    #     print(name)

    # 训练时冻结其他层，只训练head层，用于迁移学习
    if freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)

    # 使用adam优化器（默认SGD）
    params = [p for p in model.parameters() if p.requires_grad]
    if use_adam:
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=eps)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # 自定义学习速率变化
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if not evolve:  # 训练一次
        best_acc = 0
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)

            scheduler.step()

            # 验证
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

            # tensorboard
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            # 保存 best model 和 last model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))
            torch.save(model.state_dict(), os.path.join(save_path, "last.pth"))

    else:  # 网格法训练找到最优超参数
        logger.info("Training hyperparameters: lr={}, lrf={}, momentum={}, eps={}, weight_decay={}".format(lr, lrf,
                                                                                                           momentum,
                                                                                                           eps,
                                                                                                           weight_decay))
        best_acc = 0
        val_acc = 0

        now = time.strftime("____%H_%M_%S", time.localtime(time.time()))
        save_path = os.path.join(save_path, "lr-{}_lrf-{}".format(lr, lrf) + now)
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)

            scheduler.step()

            # 验证
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

            # tensorboard
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            # save the best model and the last model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))
            torch.save(model.state_dict(), os.path.join(save_path, "last.pth"))

        logger.info("lr={}, lrf={}, val_acc:{}".format(lr, lrf, val_acc))

        if best_acc > last_best_acc:
            logger.info("Now the best hyperparameters is lr={}, lrf={}. get val_acc={}".format(lr, lrf, val_acc))
            return best_acc

        return last_best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu, default=0(only one gpu)')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, default=16')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-08)
    parser.add_argument('--ops', type=float, default=0.1)
    parser.add_argument('--freeze-layers', action='store_true', help='freeze conv layers')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=1, help='maximum number of dataloader workers, default=1')
    parser.add_argument('--data-path', type=str, default="E:/dataset/FLOWER/flower_photos", help="dataset path")
    parser.add_argument('--save-path', type=str, default='weights', help='save path folder')

    opt = parser.parse_args()

    # log info
    set_logging()

    # train
    if not opt.evolve:
        # 直接训练
        result = train(opt)
    else:
        # 超参数随机组合
        meta = {'lr': [1e-1, 1e-3, 1e-5],  # or list(np.logspace(np.log10(0.0001), np.log10(0.1), base = 10, num = 10))
                'lrf': [1, 0.1],
                'momentum': [1e-4, ],
                'eps': [0.99, ],
                'weight_decay': [1e-8, ]
                }
        best_acc = 0
        hyperparameters = dict()
        for _ in range(10):
            for k, v in meta.items():
                for k, v in meta.items():
                    hyperparameters[k] = random.choice(v)

            opt.lr = hyperparameters['lr']
            opt.lrf = hyperparameters['lrf']
            opt.momentum = hyperparameters['momentum']
            opt.eps = hyperparameters['eps']
            opt.weight_decay = hyperparameters['weight_decay']

            best_acc = train(opt, best_acc)
