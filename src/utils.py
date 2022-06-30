import random, logging, os, json
from os.path import join

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig

from arguments import get_args
from global_variables import *


def seed_all(seed=None):
    """ Set random seed number """
    if seed is None:
        args = get_args()
        if args.seed > 0:
            seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def logger_level(args=None) -> int:
    if args is None:
        args = get_args()

    if args.debug:
        return logging.DEBUG
    return logging.INFO

def get_logger(args=None):
    """ Get logger based on args """
    if args is None:
        args = get_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logger_level(args))
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    stream_hander = logging.StreamHandler()	
    stream_hander.setFormatter(formatter)	
    logger.addHandler(stream_hander)
    transformers.logging.set_verbosity_warning()
    return logger

def get_device(args=None):
    """ Get device based on args """ 
    if args is None:
        args = get_args()

    device = "cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu"
    if device == 'cuda':
        try:
            cuda_devices = 'cuda' + os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError:
            cuda_devices = 'cuda'
    else:
        cuda_devices = 'cpu'
    return device, cuda_devices

def get_tokenizer(args=None):
    if args is None:
        args = get_args()

    plm_name = args.plm
    return AutoTokenizer.from_pretrained(plm_name)

def get_config(args=None):
    if args is None:
        args = get_args()

    config = AutoConfig.from_pretrained(args.plm)
    config.plm_name = args.plm
    config.debug = args.debug
    config.task = args.task
    config.label_type = args.label_type

    # Model
    config.tokenizer = get_tokenizer(args)
    config.num_graph_convs = args.num_graph_convs
    config.prefix_len = args.prefix_len
    config.prefix_projection = True
    config.prefix_hidden_size = args.prefix_hidden_size
    config.hidden_dropout_prob = 0.1
    # config.num_labels = len(Label)
    config.problem_type = args.problem_type

    # Train
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.max_epoch = args.max_epoch
    config.device, config.cuda_devices = get_device(args)
    config.name = args.name

    # Dataset
    config.num_name = int(args.name[0])
    config.num_attr = int(args.name[1])
    config.num_single_rel = int(args.name[2])
    config.num_double_rel = int(args.name[3])
    config.num_most_rel = int(args.name[4])
    config.num_common_sense = int(args.name[5])
    config.num_ordinal_rel = int(args.name[6])
    
    return config

def open_json(filename):
    try:
        with open(filename) as f:
            file = json.load(f)
    except FileNotFoundError:
        return None
    return file

def open_dataset(dataset_filename):
    config = get_config()
    return open_json(join(join(DATA_PATH, config.task), dataset_filename))

def open_template(template_filename):
    return open_json(join(TEMPLATES_PATH, template_filename))


logger = get_logger()
config = get_config()
