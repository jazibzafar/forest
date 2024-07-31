from src.utils import get_args_parser, yaml_to_dataclass_parser_seg
from light_unet import train_unet

# output_naming_convention: dino_vit_t/s_f/i/r_100/200/300/400/500

if __name__ == '__main__':
    arg_parser = get_args_parser()
    arg_list = arg_parser.parse_args()
    inputs = yaml_to_dataclass_parser_seg(arg_list.yaml_file)
    train_unet(inputs)