import argparse

def get_parser(config_file=None):

    if config_file is None:
        config_file = 'scripts/configs/pytorchcv/efficientnet_b0_config.yaml'

    parser = argparse.ArgumentParser(description='PyTorch Image Classification Trainer - Ed Morris (c) 2021')

    # Get the config file for the model
    parser.add_argument(
        '--config', 
        metavar="FILE", 
        help='Path and name of configuration file for training. Should be a .yaml file.', 
        default=config_file)

    # Allow the overriding of configuration by command line KEY VALUE pairs
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

    