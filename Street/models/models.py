### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch


def create_model(opt, data_size=None):
    if opt.model == 'segAdv':
        from .segAdv_model import Pix2PixHDModel_segAdv
        model = Pix2PixHDModel_segAdv(opt)
    elif opt.model == 'pix2pixHD_condImg':
        from .pix2pixHD_condImg_model import Pix2PixHDModel_condImg
        model = Pix2PixHDModel_condImg(opt)
    elif opt.model == 'pix2pixHD_segAdv':
        from .pix2pixHD_segAdv_model import Pix2PixHDModel_segAdv
        model = Pix2PixHDModel_segAdv(opt)
    elif opt.model == 'pix2pixHD_detectAdv':
        from .pix2pixHD_detectAdv_model import Pix2PixHDModel_detectAdv
        model = Pix2PixHDModel_detectAdv(opt)
    elif opt.model == 'pix2pixHD_condImgColor':
        from .pix2pixHD_condImgColor_model import Pix2PixHDModel_condImgColor
        model = Pix2PixHDModel_condImgColor(opt)
    else:
        raise NotImplementedError("the model is not implemented")

    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
