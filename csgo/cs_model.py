# import torch
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import check_img_size


def load_model(args):
    device = select_device('' if args.use_cuda else 'cpu')
    model = DetectMultiBackend(args.model_path, device=device, dnn=False, fp16=False)
    # model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((args.imgsz, args.imgsz), s=stride)
    model.warmup(imgsz=(1, 3, *imgsz))
    return model, stride, names
