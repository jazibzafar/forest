import yaml
import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def yaml_to_dict_parser(file):
    with open(file, 'r') as f:
        yml_params = yaml.safe_load(f)
    return yml_params


def write_dict_to_yaml(yaml_file, dict_in):
    with open(yaml_file, 'w') as outfile:
        yaml.dump(dict_in, outfile, default_flow_style=False)


def write_dict_to_csv(file_to_write, dict_to_write):
    with open(file_to_write, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_to_write.items():
            writer.writerow([key, value])


def acc_event_to_dict(path_event, dict_in):
    ea = EventAccumulator(path_event).Reload()
    tags = ea.Tags()['scalars']

    for tag in tags:
        tag_values = []
        for event in ea.Scalars(tag):
            tag_values.append(event.value)
            # the conditional below ensures that singletons are not interpreted as lists.
            if len(tag_values) > 1:
                dict_in[tag] = tag_values
            else:
                dict_in[tag] = event.value
    return dict_in


def event_to_yml(path):
    ignore = "hparams.yaml"
    stat_dict = {}
    for dname in os.listdir(path):
        if not dname == ignore:
            path_event = os.path.join(path, dname)
            acc_event_to_dict(path_event, stat_dict)
    write_file = os.path.join(path, "stats.yaml")
    write_dict_to_yaml(write_file, stat_dict)



