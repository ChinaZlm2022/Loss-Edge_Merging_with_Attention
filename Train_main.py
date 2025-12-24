import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import data_utils as d_utils
import shapenet_part_loader
from model_main import _netlocalD, _netG

from models import EGPBWithCoordinate, IterativeDynamicSelector, MHCPB, IterativeEnhancementPipeline
from utils import process_edge_features, process_with_egpb, process_with_mhcpb
from config import get_config
import utils
import random
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def setup_models(opt, device):
    """设置所有模型"""
    point_netG = _netG(opt.num_scales, opt.each_scales_size, opt.point_scales_list, opt.crop_point_num)
    point_netD = _netlocalD(opt.crop_point_num)

    # 初始化自定义模块
    egpb_processor = EGPBWithCoordinate(
        feature_channels=opt.egpb_channels,
        coord_channels=3,
        hidden_channels=128,
        num_heads=4,
        dropout=opt.drop
    )

    ids_selector = IterativeDynamicSelector(
        feature_dim=opt.egpb_channels,
        num_selected=opt.ids_selected_num,
        num_iterations=3
    )

    mhcpb_processor = MHCPB(
        d_model=opt.mhcpb_channels,
        nhead=4,
        dim_feedforward=1024,
        dropout=opt.drop
    )

    # 新增：Iterative Enhancement Pipeline
    iep_processor = IterativeEnhancementPipeline(
        d_model=opt.mhcpb_channels,
        nhead=4,
        egpb_channels=opt.egpb_channels,
        num_iterations=opt.iep_iterations,
        dropout=opt.drop
    )

    # 移动到设备
    models = {
        'point_netG': point_netG.to(device),
        'point_netD': point_netD.to(device),
        'egpb_processor': egpb_processor.to(device),
        'ids_selector': ids_selector.to(device),
        'mhcpb_processor': mhcpb_processor.to(device),
        'iep_processor': iep_processor.to(device)  # 新增
    }

    # 初始化权重
    for model in models.values():
        model.apply(weights_init_normal)

    return models


def setup_data_loaders(opt):
    """设置数据加载器"""
    transforms = d_utils.PointcloudToTensor()

    dset = shapenet_part_loader.PartDataset(
        root='/root/autodl-zm/dataset/pcn_dataset', which_dataset="PCN", classification=True,
        class_choice=None, npoints=opt.pnum, split='train')

    test_dset = shapenet_part_loader.PartDataset(
        root='/root/autodl-zm/dataset/pcn_dataset', which_dataset="PCN", classification=True,
        class_choice=None, npoints=opt.pnum, split='test')

    dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                                  shuffle=True, num_workers=int(opt.workers))

    return dataloader, test_dataloader


def setup_optimizers(opt, models):
    """设置优化器"""
    optimizerD = torch.optim.Adam(models['point_netD'].parameters(), lr=0.0001,
                                  betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)

    optimizerG = torch.optim.Adam(
        list(models['point_netG'].parameters()) +
        list(models['egpb_processor'].parameters()) +
        list(models['ids_selector'].parameters()) +
        list(models['mhcpb_processor'].parameters()) +
        list(models['iep_processor'].parameters()),  # 新增
        lr=0.0001, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay
    )

    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

    return optimizerD, optimizerG, schedulerD, schedulerG


def process_with_iep(iep_processor, mhcpb_output, f_11, iep_channels):
    """
    使用IEP模块处理MHCPB输出和F_11
    """
    batch_size, num_points_mhcpb, feature_dim_mhcpb = mhcpb_output.shape
    _, num_points_f11, feature_dim_f11 = f_11.shape

    # 确保特征维度一致
    if feature_dim_mhcpb != iep_channels:
        projection = nn.Linear(feature_dim_mhcpb, iep_channels).to(mhcpb_output.device)
        mhcpb_output_proj = projection(mhcpb_output)
    else:
        mhcpb_output_proj = mhcpb_output

    if feature_dim_f11 != iep_channels:
        projection_f11 = nn.Linear(feature_dim_f11, iep_channels).to(f_11.device)
        f_11_proj = projection_f11(f_11)
    else:
        f_11_proj = f_11

    # 使用IEP处理
    iep_output = iep_processor(mhcpb_output_proj, f_11_proj)

    return iep_output


def train_epoch(epoch, opt, dataloader, test_dataloader, models, optimizers, criterions, device, resume_epoch=0):
    """训练一个epoch"""
    optimizerD, optimizerG = optimizers
    criterion, criterion_PointLoss = criterions

    # 获取模型
    point_netG = models['point_netG']
    point_netD = models['point_netD']
    egpb_processor = models['egpb_processor']
    ids_selector = models['ids_selector']
    mhcpb_processor = models['mhcpb_processor']
    iep_processor = models['iep_processor']

    real_label = 1
    fake_label = 0

    crop_point_num = int(opt.crop_point_num)
    num_batch = len(dataloader.dataset) / opt.batchSize

    ###########################
    #  G-NET and T-NET
    ##########################
    if opt.D_choose == 1:
        for epoch in range(resume_epoch, opt.niter):
            if epoch < 30:
                alpha1 = 0.01
                alpha2 = 0.02
            elif epoch < 80:
                alpha1 = 0.05
                alpha2 = 0.1
            else:
                alpha1 = 0.1
                alpha2 = 0.2

            for i, data in enumerate(dataloader, 0):
                real_point, target = data
                batch_size = real_point.size()[0]
                real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
                input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
                input_cropped1 = input_cropped1.data.copy_(real_point)
                real_point = torch.unsqueeze(real_point, 1)
                input_cropped1 = torch.unsqueeze(input_cropped1, 1)
                p_origin = [0, 0, 0]

                if opt.cropmethod == 'random_center':
                    # Set viewpoints
                    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                              torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
                    for m in range(batch_size):
                        index = random.sample(choice, 1)  # Random choose one of the viewpoint
                        distance_list = []
                        p_center = index[0]
                        for n in range(opt.pnum):
                            distance_list.append(utils.distance_squre(real_point[m, 0, n], p_center))
                        distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

                        for sp in range(opt.crop_point_num):
                            input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
                            real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]

                label = torch.FloatTensor(batch_size, 1).fill_(real_label)
                real_point = real_point.to(device)
                real_center = real_center.to(device)
                input_cropped1 = input_cropped1.to(device)
                label = label.to(device)

                ############################
                # (1) data prepare
                ###########################
                real_center = Variable(real_center, requires_grad=True)
                real_center = torch.squeeze(real_center, 1)
                real_center_key1_idx = utils.farthest_point_sample(real_center, 64, RAN=False)
                real_center_key1 = utils.index_points(real_center, real_center_key1_idx)
                real_center_key1 = Variable(real_center_key1, requires_grad=True)

                real_center_key2_idx = utils.farthest_point_sample(real_center, 128, RAN=True)
                real_center_key2 = utils.index_points(real_center, real_center_key2_idx)
                real_center_key2 = Variable(real_center_key2, requires_grad=True)

                input_cropped1 = torch.squeeze(input_cropped1, 1)
                input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
                input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
                input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
                input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)
                input_cropped1 = Variable(input_cropped1, requires_grad=True)
                input_cropped2 = Variable(input_cropped2, requires_grad=True)
                input_cropped3 = Variable(input_cropped3, requires_grad=True)
                input_cropped2 = input_cropped2.to(device)
                input_cropped3 = input_cropped3.to(device)
                b = input_cropped3.shape[0]

                # 处理边缘特征
                edge_features = process_edge_features(input_cropped3, batch_size)
                edge_features = edge_features.to(device)

                # 使用EGPB处理input_cropped3和边缘特征
                enhanced_cropped3 = process_with_egpb(egpb_processor, input_cropped3, edge_features)

                # 使用IDS从input_cropped2中筛选特征
                input_cropped2_coords = input_cropped2.transpose(1, 2)
                input_cropped2_features = egpb_processor(input_cropped2_coords, input_cropped2_coords)
                f_ids, f_ids_indices = ids_selector(input_cropped2_features, input_cropped2_coords)
                f_ids = f_ids.transpose(1, 2)

                # 使用MHCPB处理EGPB输出和F_ids
                mhcpb_output = process_with_mhcpb(mhcpb_processor, enhanced_cropped3, f_ids, opt.mhcpb_channels)

                # ========== 新增部分：Iterative Enhancement Pipeline ==========
                # 使用IDS从input_cropped1生成F_11
                input_cropped1_coords = input_cropped1.transpose(1, 2)
                input_cropped1_features = egpb_processor(input_cropped1_coords, input_cropped1_coords)
                f_11, f_11_indices = ids_selector(input_cropped1_features, input_cropped1_coords)
                f_11 = f_11.transpose(1, 2)

                # 将MHCPB输出和F_11输入到IEP中
                iep_output = process_with_iep(iep_processor, mhcpb_output, f_11, opt.mhcpb_channels)

                # 使用IEP输出作为最终的input_cropped3
                input_cropped3_final = iep_output
                # ========== 新增部分结束 ==========

                # 构建最终的input_cropped
                input_cropped = [input_cropped1, input_cropped2, input_cropped3_final]

                point_netG = point_netG.train()
                point_netD = point_netD.train()

                ############################
                # (2) Update D network
                ###########################
                point_netD.zero_grad()
                real_center_unsqueezed = torch.unsqueeze(real_center, 1)
                output = point_netD(real_center_unsqueezed)
                errD_real = criterion(output, label)
                errD_real.backward()

                # 生成假样本
                fake_center1, fake_center2, fake = point_netG(input_cropped, edge_features)
                fake = torch.unsqueeze(fake, 1)

                label.data.fill_(fake_label)
                output = point_netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (3) Update G network: maximize log(D(G(z)))
                ###########################
                point_netG.zero_grad()
                label.data.fill_(real_label)
                output = point_netD(fake)
                errG_D = criterion(output, label)
                errG_l2 = 0
                CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))

                errG_l2 = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1)) \
                          + alpha1 * criterion_PointLoss(fake_center1, real_center_key1) \
                          + alpha2 * criterion_PointLoss(fake_center2, real_center_key2)

                errG = (1 - opt.wtl2) * errG_D + opt.wtl2 * errG_l2
                errG.backward()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f'
                      % (epoch, opt.niter, i, len(dataloader),
                         errD.data, errG_D.data, errG_l2, errG, CD_LOSS))

                f = open('loss_main_with_iep.txt', 'a')
                f.write('\n' + '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f /%.4f'
                        % (epoch, opt.niter, i, len(dataloader),
                           errD.data, errG_D.data, errG_l2, errG, CD_LOSS))

                if i % 10 == 0:
                    print('After, ', i, '-th batch')
                    f.write('\n' + 'After, ' + str(i) + '-th batch')

                    # 测试逻辑
                    for test_i, test_data in enumerate(test_dataloader, 0):
                        real_point, target = test_data

                        batch_size = real_point.size()[0]
                        real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
                        input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
                        input_cropped1 = input_cropped1.data.copy_(real_point)
                        real_point = torch.unsqueeze(real_point, 1)
                        input_cropped1 = torch.unsqueeze(input_cropped1, 1)

                        p_origin = [0, 0, 0]

                        if opt.cropmethod == 'random_center':
                            choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                                      torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]

                            for m in range(batch_size):
                                index = random.sample(choice, 1)
                                distance_list = []
                                p_center = index[0]
                                for n in range(opt.pnum):
                                    distance_list.append(utils.distance_squre(real_point[m, 0, n], p_center))
                                distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
                                for sp in range(opt.crop_point_num):
                                    input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
                                    real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]

                        real_center = real_center.to(device)
                        real_center = torch.squeeze(real_center, 1)
                        input_cropped1 = input_cropped1.to(device)
                        input_cropped1 = torch.squeeze(input_cropped1, 1)
                        input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1],
                                                                         RAN=True)
                        input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
                        input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2],
                                                                         RAN=False)
                        input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)

                        # 测试时也使用完整的处理流程（包括IEP）
                        edge_features_test = process_edge_features(input_cropped3, batch_size)
                        edge_features_test = edge_features_test.to(device)
                        enhanced_cropped3_test = process_with_egpb(egpb_processor, input_cropped3, edge_features_test)

                        # 使用IDS从input_cropped2中筛选特征
                        input_cropped2_coords_test = input_cropped2.transpose(1, 2)
                        input_cropped2_features_test = egpb_processor(input_cropped2_coords_test,
                                                                      input_cropped2_coords_test)
                        f_ids_test, _ = ids_selector(input_cropped2_features_test, input_cropped2_coords_test)
                        f_ids_test = f_ids_test.transpose(1, 2)

                        # 使用MHCPB处理
                        mhcpb_output_test = process_with_mhcpb(mhcpb_processor, enhanced_cropped3_test, f_ids_test,
                                                               opt.mhcpb_channels)

                        # 使用IDS从input_cropped1生成F_11
                        input_cropped1_coords_test = input_cropped1.transpose(1, 2)
                        input_cropped1_features_test = egpb_processor(input_cropped1_coords_test,
                                                                      input_cropped1_coords_test)
                        f_11_test, _ = ids_selector(input_cropped1_features_test, input_cropped1_coords_test)
                        f_11_test = f_11_test.transpose(1, 2)

                        # 使用IEP处理
                        iep_output_test = process_with_iep(iep_processor, mhcpb_output_test, f_11_test,
                                                           opt.mhcpb_channels)

                        input_cropped1 = Variable(input_cropped1, requires_grad=False)
                        input_cropped2 = Variable(input_cropped2, requires_grad=False)
                        input_cropped3 = Variable(input_cropped3, requires_grad=False)
                        input_cropped2 = input_cropped2.to(device)
                        input_cropped3 = input_cropped3.to(device)

                        input_cropped_test = [input_cropped1, input_cropped2, iep_output_test]
                        point_netG.eval()
                        fake_center1, fake_center2, fake = point_netG(input_cropped_test, edge_features_test)
                        CD_loss = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))
                        print('test result with IEP:', CD_loss)
                        f.write('\n' + 'test result with IEP:  %.4f' % (CD_loss))
                        break
                f.close()

            schedulerD.step()
            schedulerG.step()

            if epoch % 10 == 0:
                torch.save({'epoch': epoch + 1,
                            'state_dict': point_netG.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/point_netG_with_iep' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': point_netD.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/point_netD_with_iep' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': egpb_processor.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/egpb_processor' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': ids_selector.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/ids_selector' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': mhcpb_processor.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/mhcpb_processor' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': iep_processor.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/iep_processor' + str(epoch) + '.pth')



    #
    #############################
    ## ONLY G-NET
    ############################
    else:
        for epoch in range(resume_epoch, opt.niter):
            if epoch < 30:
                alpha1 = 0.01
                alpha2 = 0.02
            elif epoch < 80:
                alpha1 = 0.05
                alpha2 = 0.1
            else:
                alpha1 = 0.1
                alpha2 = 0.2

            for i, data in enumerate(dataloader, 0):
                real_point, target = data
                batch_size = real_point.size()[0]
                real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
                input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
                input_cropped1 = input_cropped1.data.copy_(real_point)
                real_point = torch.unsqueeze(real_point, 1)
                input_cropped1 = torch.unsqueeze(input_cropped1, 1)
                p_origin = [0, 0, 0]

                if opt.cropmethod == 'random_center':
                    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                              torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
                    for m in range(batch_size):
                        index = random.sample(choice, 1)
                        distance_list = []
                        p_center = index[0]
                        for n in range(opt.pnum):
                            distance_list.append(utils.distance_squre(real_point[m, 0, n], p_center))
                        distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

                        for sp in range(opt.crop_point_num):
                            input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
                            real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]

                real_point = real_point.to(device)
                real_center = real_center.to(device)
                input_cropped1 = input_cropped1.to(device)

                ############################
                # (1) data prepare
                ###########################
                real_center = Variable(real_center, requires_grad=True)
                real_center = torch.squeeze(real_center, 1)
                real_center_key1_idx = utils.farthest_point_sample(real_center, 64, RAN=False)
                real_center_key1 = utils.index_points(real_center, real_center_key1_idx)
                real_center_key1 = Variable(real_center_key1, requires_grad=True)

                real_center_key2_idx = utils.farthest_point_sample(real_center, 128, RAN=True)
                real_center_key2 = utils.index_points(real_center, real_center_key2_idx)
                real_center_key2 = Variable(real_center_key2, requires_grad=True)

                input_cropped1 = torch.squeeze(input_cropped1, 1)
                input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
                input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
                input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
                input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)

                # 处理边缘特征
                edge_features = process_edge_features(input_cropped3, batch_size)
                edge_features = edge_features.to(device)

                # 使用EGPB处理input_cropped3和边缘特征
                enhanced_cropped3 = process_with_egpb(egpb_processor, input_cropped3, edge_features)

                # 使用IDS从input_cropped2中筛选特征
                input_cropped2_coords = input_cropped2.transpose(1, 2)
                input_cropped2_features = egpb_processor(input_cropped2_coords, input_cropped2_coords)
                f_ids, f_ids_indices = ids_selector(input_cropped2_features, input_cropped2_coords)
                f_ids = f_ids.transpose(1, 2)

                # 使用MHCPB处理EGPB输出和F_ids
                mhcpb_output = process_with_mhcpb(mhcpb_processor, enhanced_cropped3, f_ids, opt.mhcpb_channels)

                # ========== 新增部分：Iterative Enhancement Pipeline ==========
                # 使用IDS从input_cropped1生成F_11
                input_cropped1_coords = input_cropped1.transpose(1, 2)
                input_cropped1_features = egpb_processor(input_cropped1_coords, input_cropped1_coords)
                f_11, f_11_indices = ids_selector(input_cropped1_features, input_cropped1_coords)
                f_11 = f_11.transpose(1, 2)

                # 将MHCPB输出和F_11输入到IEP中
                iep_output = process_with_iep(iep_processor, mhcpb_output, f_11, opt.mhcpb_channels)

                # 使用IEP输出作为最终的input_cropped3
                input_cropped3_final = iep_output
                # ========== 新增部分结束 ==========

                input_cropped1 = Variable(input_cropped1, requires_grad=True)
                input_cropped2 = Variable(input_cropped2, requires_grad=True)
                input_cropped3 = Variable(input_cropped3, requires_grad=True)
                input_cropped2 = input_cropped2.to(device)
                input_cropped3 = input_cropped3.to(device)

                # 使用增强后的特征
                input_cropped = [input_cropped1, input_cropped2, input_cropped3_final]

                point_netG = point_netG.train()
                point_netG.zero_grad()
                fake_center1, fake_center2, fake = point_netG(input_cropped, edge_features)
                fake = torch.unsqueeze(fake, 1)

                ############################
                # (3) Update G network: maximize log(D(G(z)))
                ###########################

                CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))

                errG_l2 = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1)) \
                          + alpha1 * criterion_PointLoss(fake_center1, real_center_key1) \
                          + alpha2 * criterion_PointLoss(fake_center2, real_center_key2)

                errG_l2.backward()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                      % (epoch, opt.niter, i, len(dataloader),
                         errG_l2, CD_LOSS))

                f = open('loss_main_with_iep.txt', 'a')
                f.write('\n' + '[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                        % (epoch, opt.niter, i, len(dataloader),
                           errG_l2, CD_LOSS))
                f.close()

            schedulerD.step()
            schedulerG.step()

            if epoch % 10 == 0:
                torch.save({'epoch': epoch + 1,
                            'state_dict': point_netG.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/point_netG_with_iep' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': egpb_processor.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/egpb_processor' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': ids_selector.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/ids_selector' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': mhcpb_processor.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/mhcpb_processor' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': iep_processor.state_dict()},
                           '/root/autodl-zm/zm-test01/Trained_Model/iep_processor' + str(epoch) + '.pth')


def save_models(epoch, models):
    """保存所有模型"""
    for name, model in models.items():
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict()
        }, f'/root/autodl-zm/zm-test01/Trained_Model/{name}_{epoch}.pth')


def main():
    opt = get_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    # 设置模型
    models = setup_models(opt, device)

    # 设置数据加载器
    dataloader, test_dataloader = setup_data_loaders(opt)

    # 设置优化器
    optimizerD, optimizerG, schedulerD, schedulerG = setup_optimizers(opt, models)

    # 设置损失函数
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_PointLoss = utils.PointLoss().to(device)

    # 训练循环
    resume_epoch = 0
    if opt.netG != '':
        models['point_netG'].load_state_dict(
            torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.netG)['epoch']
    if opt.netD != '':
        models['point_netD'].load_state_dict(
            torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.netD)['epoch']

    for epoch in range(resume_epoch, opt.niter):
        train_epoch(epoch, opt, dataloader, test_dataloader, models,
                    (optimizerD, optimizerG),
                    (criterion, criterion_PointLoss), device, resume_epoch)

        schedulerD.step()
        schedulerG.step()

        # 保存模型
        if epoch % 10 == 0:
            save_models(epoch, models)


if __name__ == '__main__':
    main()