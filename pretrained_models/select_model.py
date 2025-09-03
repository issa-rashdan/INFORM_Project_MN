
def select_model(args):
    if args.arch[:3] == "vit":
        from .vit.load_vit_model import load_vit_model
        model = load_vit_model(args)
    '''
    elif args.arch[:3] == "cnn":
    '''
    return model
    