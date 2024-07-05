from src.arg_parser import get_args_parser, yaml_to_dataclass_parser_seg
from src.light_segmentation import train_segmentation

# output_naming_convention: dino_vit_t/s_f/i/r_100/200/300/400/500

if __name__ == '__main__':
    arg_parser = get_args_parser()
    arg_list = arg_parser.parse_args()
    inputs = yaml_to_dataclass_parser_seg(arg_list.yaml_file)
    train_segmentation(inputs)
    # for input_idx in range(len(input_list)):
    #     print("current input: ")
    #     print(input_list[input_idx])
    #     train_classification(input_list[input_idx])