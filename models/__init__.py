"""Model construction entry points and registration utilities, exposing creation by name. Notable functions include: create_model."""
# Purpose: Factory: create and configure a model instance by name.
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