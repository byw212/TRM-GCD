import argparse
from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import vision_transformer as vits
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from torch.utils.data import Subset
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path,pretrain_path
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, \
    get_params_groups
from loguru import logger

def knn_cluster_Loss(feature,vector,Adjacency,prototype):
    feature = torch.nn.functional.normalize(feature, dim=-1)
    prototype = torch.nn.functional.normalize(prototype, dim=-1)
    v1,_ = vector.chunk(2)
    f1,f2 = feature.chunk(2)

    size= f1.shape[0]
    similarity_matrix = f1@f2.T
    prototype_similarity = v1@prototype.detach().T
    prototype_anchor = torch.diag(prototype_similarity)
    LARloss = torch.pow(1-prototype_anchor,2)
    anchor = torch.zeros(size).to(device)
    sub = torch.zeros(size).to(device)
    for i in range(size):
        anchor[i] = torch.exp(similarity_matrix[i]/0.35)[Adjacency[i]==1].sum()
        sub[i] = torch.exp(similarity_matrix[i]/0.25)[Adjacency[i] == -1].sum()

    PCLloss = -torch.log(anchor/sub)


    return PCLloss.mean() + LARloss.mean()


def get_Adjacency(model,dataset,K,th):
    length = len(dataset)
    data_iter = iter(dataset)
    with torch.no_grad():
        x, y, idx, z = next(data_iter)
        z = z[:, 0]
        uq_idx = idx.to(device)
        label = y.to(device)
        mask = z.to(device).bool()
        feature = model(x.to(device))[-1]
        for i in range(1, length):
            x, y, idx, z = next(data_iter)
            z = z[:, 0].bool()
            feature = torch.cat([feature, model(x.to(device))[-1]], dim=0)
            uq_idx = torch.cat([uq_idx,idx.to(device)], dim=0)
            label = torch.cat([label, y.to(device)], dim=0)
            mask = torch.cat([mask, z.to(device)], dim=0)
        sort_id,index = uq_idx.sort()
        transform = {int(i):num for num,i in enumerate(sort_id)}
        feature = feature[index]
        label = label[index]
        mask = mask[index]
        feature = torch.nn.functional.normalize(feature, dim=-1)
        size = feature.shape[0]
        similarity_matrix = feature@feature.T
        _, indice1 = similarity_matrix.sort(axis=-1)
        _, indice2 = (similarity_matrix-torch.eye(size).to(device)).sort(axis=-1)
        #similarity_matrix[:,mask] = 1
        #_, indice3 = similarity_matrix.sort(axis=-1)
    Adjacency = torch.zeros((size, size)).to(device)
    th = int(size * th)
    for i in range(size):
        Adjacency[i][indice1[i][:th]] = -1

    Adjacency_matrix = torch.zeros((size, size)).to(device)
    for i in range(size):
        Adjacency_matrix[i][indice2[i][-2:]] = 1
    mask_label = deepcopy(label)
    mask_label[~mask] = -1
    choice = mask_label.unsqueeze(0) == mask_label.unsqueeze(1)
    def matrix_pow(matrix, num):
        result = torch.eye(matrix.shape[0]).to(device)
        for i in range(num + 1):
            result = result @ matrix
        return result

    Adjacency_bool = torch.zeros((size, size)).to(device)
    for i in range(K):
        Adjacency_bool += matrix_pow(Adjacency_matrix, i)

    #Adjacency_bool += Adjacency_bool.T
    Adjacency += Adjacency_bool.bool().float()#*Adjacency_bool.bool().float().T
    prototypes = torch.zeros(feature.shape).to(device)
    for num, i in enumerate(mask_label):
        Adjacency[num][num] = 1
        if i == -1:
            continue
        Adjacency[num][choice[num]] = 1
        choice[num][~mask] = True
        Adjacency[num][~choice[num]] = -1

    for i in range(size):
        prototypes[i]=  feature[Adjacency[i]==1].mean(0)
    return Adjacency,transform,prototypes

def train_stage(student,init_model,init_dataset, train_dataset, unlabelled_train_loader, args):
    best_train_acc_lab = 0
    best_train_acc_ubl = 0
    best_train_acc_all = 0
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    length = len(train_dataset)
    for epoch in range(args.epochs):
        indices = list(range(length))
        loss_record = AverageMeter()

        student.train()
        while len(indices)!=0:
            if len(indices)<args.length_subset:
                samples = np.random.choice(indices, size=len(indices), replace=False)
            else:
                samples = np.random.choice(indices, size=args.length_subset,replace=False)
            indices = list(set(indices) - set(samples))
            sub_train_dataset = Subset(train_dataset, samples)
            sub_train_dataset2 = Subset(init_dataset, samples)
            sample_weights = [1 if i < label_len else args.w_u for i in samples]
            sample_weights = torch.DoubleTensor(sample_weights)
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(samples))

            # --------------------
            # DATALOADERS
            # --------------------

            train_loader = DataLoader(sub_train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                                      sampler=sampler,drop_last=True)
            init_loader = DataLoader(sub_train_dataset2, num_workers=args.num_workers, batch_size=256, shuffle=False,
                                       drop_last=False)

            if epoch < 10:
                Adjacency, transform,prototypes = get_Adjacency(init_model,init_loader,args.K,args.threshold)
            else:
                Adjacency, transform, prototypes = get_Adjacency(student, init_loader, args.K, args.threshold)
            for batch_idx, batch in enumerate(train_loader):
                images, class_labels, uq_idxs, mask_lab = batch
                mask_lab = mask_lab[:, 0]
                uq_idxs = torch.tensor([transform[int(i)] for i in uq_idxs])

                #class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
                class_labels, mask_lab, uq_idxs = class_labels.to(device), mask_lab.to(device).bool(), uq_idxs.to(device)
                images = torch.cat(images, dim=0).to(device)
                sub_Adjacency = Adjacency[uq_idxs]
                sub_Adjacency = sub_Adjacency[:, uq_idxs]
                prototype = prototypes[uq_idxs]

                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    student_proj, student_out,x = student(images)

                    PCL_LAR_loss = knn_cluster_Loss(student_proj,x, sub_Adjacency,prototype)
                    teacher_out = student_out.detach()

                    # clustering, sup
                    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                    # clustering, unsup
                    cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                    me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))

                    cluster_loss += args.memax_weight * me_max_loss

                    pstr = ''
                    pstr += f'cls_loss: {cls_loss.item():.4f} '
                    pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                    pstr += f'PCL_LAR_loss: {PCL_LAR_loss.item():.4f} '

                    loss = 0
                    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                    loss += PCL_LAR_loss

                # Train acc

                loss_record.update(loss.item(), class_labels.size(0))
                optimizer.zero_grad()
                if fp16_scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    fp16_scaler.scale(loss).backward()
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()

                if batch_idx % args.print_freq == 0:
                    args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                     .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch,
                                         save_name='Train ACC Unlabelled', args=args)

        args.logger.info('Stage2 Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))
        # if new_acc > best_train_acc_all:
        #
        #     args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        #
        #     torch.save(save_dict, args.model_path[:-3] + f'stage2_best.pt')
        #     args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'stage2_best.pt'))
        #
        #     # transductive
        #     best_train_acc_lab = old_acc
        #     best_train_acc_ubl = new_acc
        #     best_train_acc_all = all_acc
        #
        # args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model,test_loader, epoch, save_name, args):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            _, logits,_= model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask,
                             np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cub',
                        help='options: cifar10, cifar100, imagenet_100, cub, scars, aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=1.0)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float,
                        help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='myself', type=str)
    parser.add_argument('--length_subset', default=10000, type=int)
    parser.add_argument('--threshold', default=0.95, type=float)
    parser.add_argument('--w_u', default=0.5, type=float)
    parser.add_argument('--K', default=15, type=int)

    # ----------------------
    # ----------------------
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['TRM'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = vits.__dict__['vit_base']()
    init_backbone = vits.__dict__['vit_base']()

    state_dict = torch.load(dino_pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    init_dataset = deepcopy(train_dataset)
    init_dataset.unlabelled_dataset.transform = test_transform
    init_dataset.labelled_dataset.transform = test_transform
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    init_projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    pretrain_path += args.dataset_name + '.pt'
    state_dict = torch.load(pretrain_path, map_location='cpu')
    init_model = nn.Sequential(init_backbone, init_projector).to(device)
    init_model.load_state_dict(state_dict['model'])
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    train_stage(model,init_model, init_dataset,train_dataset, test_loader_unlabelled, args)
