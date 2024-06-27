import yaml
import argparse
from src.parameters import ClassificationParameters


def yaml_to_dict_parser(file):
    with open(file, 'r') as f:
        args = yaml.safe_load(f)
    return args


def yaml_to_dataclass_parser(file):
    arg_dict = yaml_to_dict_parser(file)
    arg_dataclass = ClassificationParameters(**arg_dict)
    return arg_dataclass


def yaml_to_dataclass_sequential(file):
    dataclass_list = []
    arg_dict = yaml_to_dict_parser(file)
    assert len(arg_dict['checkpoint_path']) == len(arg_dict['output_dir'])
    for i in range(len(arg_dict['checkpoint_path'])):
        dt_cls = ClassificationParameters(arch=arg_dict['arch'],
                                          checkpoint_path=arg_dict['checkpoint_path'][i],
                                          data_path=arg_dict['data_path'],
                                          output_dir=arg_dict['output_dir'][i],
                                          resume_ckpt=arg_dict['resume_ckpt'])
        dataclass_list.append(dt_cls)
    return dataclass_list


def get_args_parser():
    parser = argparse.ArgumentParser('PARSE YAML', add_help=False)
    parser.add_argument('--yaml_file', type=str, required=True,
                        help="path to the yaml file containing parameters.")
    return parser
