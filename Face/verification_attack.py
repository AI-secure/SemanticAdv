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

    save_path = config.save_path
    threshold = config.threshold
    if threshold > 0 and config.untargeted == True:
        threshold = -threshold
    if config.test_threshold == 0:
        test_threshold = threshold
    else:
        test_threshold = config.test_threshold

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
                                         threshold / 256)

    # Read images
    test_idlist = []
    test_namelist = []
    if config.data_mode == 'demo':
        sub_folder = ''
        data_list = [0, 2, 10, 24, 28, 69]
        if config.untargeted == True:
            original_list = [0, 1, 2, 3, 4, 5]
        else:
            original_list = [1, 2, 3, 4, 5]
    elif config.data_mode == 'all':
        sub_folder = 'ori/'
        data_list = range(5000)
        original_list = range(1280)

    for temp_i in data_list:
        temp_path1 = config.celeba_image_dir + str(temp_i + 1) + '/' + sub_folder
        test_idlist.append(str(temp_i + 1))
        test_namelist.append(sorted(os.listdir(temp_path1))[0])

    dic_label, dic_image = create_dic(config.celeba_image_dir,
                                      config.attr_path, config.selected_attrs,
                                      config.celeba_crop_size,
                                      config.image_size, test_idlist,
                                      test_namelist, sub_folder)

    dic_attribute = dict(zip(['Blond_Hair', 'Wavy_Hair', 'Young', 'Eyeglasses', 'Heavy_Makeup', 'Rosy_Cheeks',
                              'Chubby', 'Mouth_Slightly_Open', 'Bushy_Eyebrows', 'Wearing_Lipstick', 'Smiling',
                              'Arched_Eyebrows', 'Bangs', 'Wearing_Earrings', 'Bags_Under_Eyes', 'Receding_Hairline', 'Pale_Skin'],
                             range(17)))

    for index in original_list:
        if config.untargeted == True:
            i_target = index
        else:
            if config.data_mode == 'demo':
                i_target = 0
            elif config.data_mode == 'all':
                i_target = index + 2000

        t_x_real = dic_image[test_namelist[i_target]]
        t_x_real = t_x_real.unsqueeze(0)
        t_img_ori = rec_transform(denorm1(t_x_real)).cuda()

        t_ori_face_feature = model(t_img_ori)
        t_face_feature_const = torch.zeros_like(t_ori_face_feature)
        t_face_feature_const.data = t_ori_face_feature.clone()
        t_face_feature_const = t_face_feature_const / torch.norm(
            t_face_feature_const, dim=1)

        save_image(
            save_path + str(index) + '_' + str(i_target) + '_' +
            'target_img.png', denorm1(t_x_real))

        c_org = dic_label[test_namelist[index]]
        c_org = c_org.unsqueeze(0)
        x_real = dic_image[test_namelist[index]]
        x_real = x_real.unsqueeze(0)
        c_org = c_org.cuda()
        x_real = x_real.cuda()
        x_real_constant = denorm1(x_real).clone().cuda()
        save_image(
            save_path + str(index) + '_' + str(i_target) + '_' +
            'original_img.png', x_real_constant)

        delta = torch.zeros_like(c_org)
        delta = delta.cuda()
        x_real.requires_grad = True
        optimizer = optim.Adam([x_real], lr=0.01)

        # Opitimize X to get X'. G(X',c) looks more similar than G(X,c).
        for z in range(300):
            denormed_adv = solver.enc(delta, x_real, c_org)
            edit_final = solver.dec(denormed_adv)

            img_loss = criterionL2(edit_final, x_real_constant)
            face_loss = criterionL2(denorm1(x_real), x_real_constant)
            loss = img_loss + face_loss * 1.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            x_real.data = torch.clamp(x_real.data, -1, 1)

        new_x_real = x_real.clone().cuda()

        with torch.no_grad():
            generated_ori = solver.enc(delta, new_x_real, c_org)

        if config.adv_attribute == 'all':
            attribute_list = range(17)
        else:
            attribute_list = [dic_attribute[config.adv_attribute]]

        for j in attribute_list:
            delta = torch.zeros_like(c_org)
            delta[:, j] = delta[:, j] + 1
            delta = delta.cuda()

            with torch.no_grad():
                denormed_adv = solver.enc(delta, new_x_real, c_org)

            edit_final, adv_loss, tv_loss = adversary(
                G_dec=solver.dec,
                emb1=generated_ori,
                emb2=denormed_adv,
                model=end_model,
                loss_func=criterionL2,
                target_label=t_face_feature_const,
                targeted=(not config.untargeted))
            adv_dist = adv_loss.item() * 256
            tv_dist = tv_loss.item()

            print('source id:', index, ', target id:', i_target,
                  ', attribute index:', j, ', feature distance:', adv_dist,
                  ', attack result:', adv_dist < test_threshold)

            save_image(
                save_path + str(index) + '_' + str(i_target) + '_' +
                str(adv_dist < test_threshold) + '_' + 'adv_G(X,c' + str(j) +
                ').png', edit_final)

            if adv_dist < test_threshold:
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

    if not osp.exists('./' + save_path):
        os.makedirs('./' + save_path)
    f = open(save_path + 'record.txt', 'w')

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
                        choices=['CelebA'],
                        help='which dataset to use, only support CelebA currently')
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
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    # Directories.
    parser.add_argument('--celeba_image_dir',
                        type=str,
                        default='./aligned_id_divied_imgs/',
                        help='path of face images')
    parser.add_argument('--attr_path',
                        type=str,
                        default='./list_attr_celeba.txt',
                        help='path of face attributes')
    parser.add_argument('--model_save_dir',
                        type=str,
                        default='./pretrain_models/',
                        help='path of pretrained stargan model')
    # Face Recognition
    parser.add_argument('--feature_dim', default=256, type=int,
                        help='feature dimensions for face verification')
    parser.add_argument('--load_path',
                        type=str,
                        default='./pretrain_models/res101_softmax.pth.tar',
                        help='path of pretrained face verification model')
    # Please don't change above setting.

    # You can change below setting.
    parser.add_argument('--max_iteration', type=int, default=200,
                        help='maximum iterations')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--tv_lambda', type=float, default=0.01, help='lambda for tv loss')
    parser.add_argument('--threshold', type=float, default=1.244,
                        help='threshold for face verification, 1.244 for fpr=10e-3，0.597 for fpr=10e-4')

    parser.add_argument('--save_path', type=str, default='results/', help='path to save the results')
    parser.add_argument('--interp_layer', type=str, default='0', choices=['0', '1', '2', '01', '02'], help='which layer to interpolate')
    parser.add_argument('--test_threshold', type=float, default=0)
    parser.add_argument('--untargeted', action='store_true', help='targeted or untargeted')
    parser.add_argument('--data_mode', type=str, default='demo', choices=['demo', 'all'], help='demo mode for simple demo, all mode for paper reproduction')
    parser.add_argument('--adv_attribute', type=str, default='all', choices=[
                            'Blond_Hair', 'Wavy_Hair', 'Young', 'Eyeglasses',
                            'Heavy_Makeup', 'Rosy_Cheeks', 'Chubby',
                            'Mouth_Slightly_Open', 'Bushy_Eyebrows',
                            'Wearing_Lipstick', 'Smiling', 'Arched_Eyebrows',
                            'Bangs', 'Wearing_Earrings', 'Bags_Under_Eyes',
                            'Receding_Hairline', 'Pale_Skin', 'all'
                        ], help='which attribute to use')



    config = parser.parse_args()
    print(config)
    main(config)