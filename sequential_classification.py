from src.arg_parser import get_args_parser, yaml_to_dataclass_sequential
from src.light_classification import train_classification

# output_naming_convention: dino_vit_t/s_f/i/r_100/200/300/400/500

if __name__ == '__main__':
    arg_parser = get_args_parser()
    args = arg_parser.parse_args()
    input_list = yaml_to_dataclass_sequential(args.yaml_file)
    for input_idx in range(len(input_list)):
        print("current input: ")
        print(input_list[input_idx])
        train_classification(input_list[input_idx])



