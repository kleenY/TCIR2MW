def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'vit_one':
        from .vit_one import ViT_model
        model = ViT_model()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
