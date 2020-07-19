import os

import torch
import torch.nn.functional as F

from stargan_model import Discriminator, Generator_dissection


class Celeba_Solver(object):
    """Solver for training and testing StarGAN."""
    def __init__(self, config):
        """Initialize configurations."""

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = 1
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.model_save_dir = config.model_save_dir

        # Build the model and tensorboard.
        self.build_model()
        # Load the trained generator.
        self.restore_model(self.test_iters)

        self.layer = config.interp_layer

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator_dissection(self.g_conv_dim, self.c_dim,
                                          self.g_repeat_num)
        elif self.dataset in ['Both']:
            self.G = Generator_dissection(
                self.g_conv_dim, self.c_dim + self.c2_dim + 2,
                self.g_repeat_num)  # 2 for mask vector.

        self.print_network(self.G, 'G')
        self.G.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print(
            'Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir,
                              '{}-G.ckpt'.format(resume_iters))
        self.G.load_state_dict(
            torch.load(G_path, map_location=lambda storage, loc: storage))

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(
                logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)
        elif dataset == 'LFW':
            return self.criterionL2(logit, F.sigmoid(target))
            # return F.binary_cross_entropy_with_logits(logit, F.sigmoid(target), size_average=False) / logit.size(0)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def construct_img(self, delta, x_real, c_org):

        c_trg = torch.clamp(c_org + delta, 0, 1)
        print(c_trg)

        # Translate images.
        adv = self.G(x_real, c_trg)
        denormed_adv = self.denorm(adv)

        return denormed_adv

    def enc(self, delta, x_real, c_org):
        c_trg = (c_org - delta * (c_org - 0.5) * 2).cuda()
        if self.layer == '0':
            feature = self.G.h0(x_real, c_trg)
        elif self.layer == '1':
            feature = self.G.h1(x_real, c_trg)
        elif self.layer == '2':
            feature = self.G.h2(x_real, c_trg)
        elif self.layer == '01':
            feature = self.G.h01(x_real, c_trg)
        elif self.layer == '02':
            feature = self.G.h02(x_real, c_trg)
        elif self.layer == '03':
            feature = self.G.h03(x_real, c_trg)

        return feature

    def dec(self, feature):

        if self.layer == '0':
            out = self.G.f0(feature)
        elif self.layer == '1':
            out = self.G.f1(feature)
        elif self.layer == '2':
            out = self.G.f2(feature)
        elif self.layer == '01':
            out = self.G.f01(feature)
        elif self.layer == '02':
            out = self.G.f02(feature)
        elif self.layer == '03':
            out = self.G.f03(feature)

        denormed_adv = self.denorm(out)

        return denormed_adv
