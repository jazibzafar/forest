from tifffile import imread


# def img_loader(path, dim_ordering="CHW"):
#     if dim_ordering == "HWC":
#         return imread(path)
#     elif dim_ordering == "CHW":
#         return imread(path).transpose((2,0,1))
#     else:
#         raise RuntimeError(f"Received wrong dim ordering '{dim_ordering}', must be either CHW or HWC.")

def img_loader(path):
    return imread(path)
