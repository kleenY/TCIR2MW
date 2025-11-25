"""Model factory and registration utilities.

This module is part of the TCIR2MW project.
Auto-generated overview (2025-11-25).

Key classes:
    None

Key functions:
    create_model

Notes:
    This module-level docstring was auto-generated. Please refine or expand as needed.
"""
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
