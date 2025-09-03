

def get_embeddings(model, x):
    # get patch embeddings from x
    # output shape: (batch_size, num_patches, projection_dim)
    x = model.prepare_tokens(x)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    cls_token = x[:, 0, :]  # CLS token embedding
    patch_tokens = x[:, 1:, :]  # patch tokens, shape (num_patches, projection_dim)
    return cls_token, patch_tokens
