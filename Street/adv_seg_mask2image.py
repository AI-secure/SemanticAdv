### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import sys

from torch.autograd import Variable

from data.data_loader import CreateDataLoader
from models.models import create_model
from options.mask2image_test_options import \
    MaskToImageTestOptions as TestOptions
from util import html
from util.visualizer import Visualizer

opt = TestOptions().parse(save=False, default_args=sys.argv[1:])
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(
    web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' %
    (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break

    generated = model.interp_attack(label=Variable(data['label']),
                                    label1=Variable(data['label1']),
                                    label2=Variable(data['label2']),
                                    inst=Variable(data['inst']),
                                    inst1=Variable(data['inst1']),
                                    image=Variable(data['image']),
                                    mask_in=Variable(data['mask_in']),
                                    mask_out=Variable(data['mask_out']),
                                    mask_target=Variable(data['mask_target']))

    visuals = model.get_current_visuals()
    visuals1 = model.get_current_visuals1()

    print('process image... %s' % ('%05d' % i))
    visualizer.save_images(webpage, visuals, ['%05d' % i])
    visualizer.save_images(webpage, visuals1, ['%05d' % i])

webpage.save()
