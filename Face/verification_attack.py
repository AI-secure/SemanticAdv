import argparse
import os
import os.path as osp
import sys

import torch
import torch.nn as nn
from torch import optim

import verification_model
from celeba_data import create_dic
from celeba_solver import Celeba_Solver
from utils import TVLoss, rec_transform, save_image

sys.path.append('../')
from attacks import semantic_attack


def denorm1(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class End_Model(nn.Module):
    def __init__(self, net):
        super(End_Model, self).__init__()
        self.net = net

    def forward(self, x):
        face_feature = self.net(rec_transform(x))
        normed_face_feature = face_feature / torch.norm(face_feature, dim=1)
        return normed_face_feature


def main(config):

    success_records = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fail_records = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    attack_records = []

    threshold = config.threshold

    solver = Celeba_Solver(config)

    model = verification_model.resnet101(feature_dim=config.feature_dim)
    model = verification_model.IdentityMapping(model)

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    verification_model.load_ckpt(config.load_path, model, strict=False)

    end_model = End_Model(model)

    criterionL2 = torch.nn.MSELoss()

    adversary = semantic_attack.FP_CW_TV(config.lr, config.max_iteration,
                                         config.tv_lambda,
                                         config.threshold / 256)

    # Read images
    test_idlist = []
    test_namelist = []
    for temp_i in [0, 2, 10, 24, 28, 69]:
        temp_path1 = config.celeba_image_dir + str(temp_i) + '/'
        test_idlist.append(str(temp_i))
        test_namelist.append(os.listdir(temp_path1)[0])

    dic_label, dic_image = create_dic(config.celeba_image_dir,
                                      config.attr_path, config.selected_attrs,
                                      config.celeba_crop_size,
                                      config.image_size, test_idlist,
                                      test_namelist)

    original_list = [1, 2, 3, 4, 5]
    target_list = [0]
    for i_target in target_list:

        t_x_real = dic_image[test_namelist[i_target]]
        t_x_real = t_x_real.unsqueeze(0)
        t_img_ori = rec_transform(denorm1(t_x_real)).cuda()

        t_ori_face_feature = model(t_img_ori)
        t_face_feature_const = torch.zeros_like(t_ori_face_feature)
        t_face_feature_const.data = t_ori_face_feature.clone()
        t_face_feature_const = t_face_feature_const / torch.norm(
            t_face_feature_const, dim=1)

        for index in original_list:
            if i_target == index:
                continue

            save_image(
                'target_results/' + str(index) + '_' + str(i_target) + '_' +
                'target_img.png', denorm1(t_x_real))

            c_org = dic_label[test_namelist[index]]
            c_org = c_org.unsqueeze(0)
            x_real = dic_image[test_namelist[index]]
            x_real = x_real.unsqueeze(0)
            c_org = c_org.cuda()
            x_real = x_real.cuda()
            x_real_constant = denorm1(x_real).clone().cuda()
            save_image(
                'target_results/' + str(index) + '_' + str(i_target) + '_' +
                'original_img.png', x_real_constant)

            delta = torch.zeros_like(c_org)
            delta = delta.cuda()
            x_real.requires_grad = True
            optimizer = optim.Adam([x_real], lr=0.01)

            # Opitimize X to get X'. G(X',c) looks more similar than G(X,c).
            for z in range(300):
                denormed_adv = solver.h0(delta, x_real, c_org)
                edit_final = solver.f0(denormed_adv)

                img_loss = criterionL2(edit_final, x_real_constant)
                face_loss = criterionL2(denorm1(x_real), x_real_constant)
                loss = img_loss + face_loss * 1.0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                x_real.data = torch.clamp(x_real.data, -1, 1)

            new_x_real = x_real.clone().cuda()

            with torch.no_grad():
                generated_ori = solver.h0(delta, new_x_real, c_org)

            for j in range(17):
                delta = torch.zeros_like(c_org)
                delta[:, j] = delta[:, j] + 1
                delta = delta.cuda()

                with torch.no_grad():
                    denormed_adv = solver.h0(delta, new_x_real, c_org)

                edit_final, adv_loss, tv_loss = adversary(
                    G_dec=solver.f0,
                    emb1=generated_ori,
                    emb2=denormed_adv,
                    model=end_model,
                    loss_func=criterionL2,
                    target_label=t_face_feature_const,
                    targeted=True)
                adv_dist = adv_loss.item() * 256
                tv_dist = tv_loss.item()

                print('source id:', index, ', target id:', i_target,
                      ', attribute index:', j, ', feature distance:', adv_dist,
                      ', attack result:', adv_dist < threshold)

                save_image(
                    'target_results/' + str(index) + '_' + str(i_target) + '_' +
                    str(adv_dist < threshold) + '_' + 'adv_G(X,c' + str(j) +
                    ').png', edit_final)

                if adv_dist < threshold:
                    attack_records.append([
                        index, i_target, j,
                        round(adv_dist, 4),
                        round(tv_dist, 5), True
                    ])
                    success_records[j] += 1

                else:
                    attack_records.append([
                        index, i_target, j,
                        round(adv_dist, 4),
                        round(tv_dist, 5), False
                    ])
                    fail_records[j] += 1

    rate_list = []
    for attr in range(17):
        if (success_records[attr] + fail_records[attr]) > 0:
            rate_list.append(success_records[attr] /
                             (success_records[attr] + fail_records[attr]))
    print('success rate for each attribute:', rate_list)

    if not osp.exists('./target_results/'):
        os.makedirs('./target_results/')
    f = open('target_results/record.txt', 'w')

    f.write(
        'source id, target id, attribute index, feature distance, tv loss, attack result'
        + '\n')

    for length in range(len(attack_records)):
        f.write(str(attack_records[length])[1:-1] + '\n')

    f.write('success rate for each attribute: ' + str(rate_list)[1:-1] + '\n')

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim',
                        type=int,
                        default=17,
                        help='dimension of domain labels (1st dataset)')
    parser.add_argument('--celeba_crop_size',
                        type=int,
                        default=112,
                        help='crop size for the CelebA dataset')
    parser.add_argument('--image_size',
                        type=int,
                        default=112,
                        help='image resolution')
    parser.add_argument('--g_conv_dim',
                        type=int,
                        default=128,
                        help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim',
                        type=int,
                        default=128,
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num',
                        type=int,
                        default=6,
                        help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num',
                        type=int,
                        default=6,
                        help='number of strided conv layers in D')

    parser.add_argument('--dataset',
                        type=str,
                        default='CelebA',
                        choices=['CelebA'])
    parser.add_argument('--c2_dim',
                        type=int,
                        default=8,
                        help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--selected_attrs',
                        '--list',
                        nargs='+',
                        help='selected attributes for the CelebA dataset',
                        default=[
                            'Blond_Hair', 'Wavy_Hair', 'Young', 'Eyeglasses',
                            'Heavy_Makeup', 'Rosy_Cheeks', 'Chubby',
                            'Mouth_Slightly_Open', 'Bushy_Eyebrows',
                            'Wearing_Lipstick', 'Smiling', 'Arched_Eyebrows',
                            'Bangs', 'Wearing_Earrings', 'Bags_Under_Eyes',
                            'Receding_Hairline', 'Pale_Skin'
                        ])
    # Test configuration.
    parser.add_argument('--test_iters',
                        type=int,
                        default=200000,
                        help='test model from this step')
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    # Directories.
    parser.add_argument('--celeba_image_dir',
                        type=str,
                        default='./aligned_id_divied_imgs/')
    parser.add_argument('--attr_path',
                        type=str,
                        default='./list_attr_celeba.txt')
    parser.add_argument('--model_save_dir',
                        type=str,
                        default='./pretrain_models/')
    # Face Recognition
    parser.add_argument('--feature_dim', default=256, type=int)
    parser.add_argument('--load_path',
                        type=str,
                        default='./pretrain_models/res101_softmax.pth.tar')
    # Please don't change above setting.

    # You can change below setting.
    parser.add_argument('--max_iteration', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--tv_lambda', type=float, default=0.01)
    parser.add_argument('--threshold', type=float, default=1.244)

    config = parser.parse_args()
    print(config)
    main(config)