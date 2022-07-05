import argparse


def get_args():
    """ Parse all the arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug', '-d',
        action='store_true'
    )
    parser.add_argument(
        '--seed',
        type=int, default=124,
        help="random seed number. set -1 if you do not want to use"
    )

    parser.add_argument(
        '--task',
        type=str, default='nav',
        help='type of task. supported: manip, nav'
    )
    parser.add_argument(
        '--label-type',
        type=str, default='color',
        help='type of label. supported: name, color'
    )

    # Dataset Generator
    parser.add_argument(
        '--gen-scene',
        action='store_true',
        help='generate scene graph dataset'
    )
    parser.add_argument(
        '--gen-text',
        type=str, default='0000000',
        help="""generate text dataset.
        1234567 means...
        1: number of text specified by name,
        2: attributes,
        3: single relations,
        4: double relations,
        5: relations includes 'most',
        6: common sense,
        7: relations with ordinal number """
    )

    # Graph Encoder
    parser.add_argument(
        '--num-graph-convs',
        type=int, default=4,
        help='number of graph convolutions i.e., number of hidden layers in the graph encoder'
    )

    # Language Model Classifier
    parser.add_argument(
        '--plm',
        type=str, default="bert-base-multilingual-uncased",
        help='pre-trained languge model to be used'
    )
    parser.add_argument(
        '--prefix-len',
        type=int, default=5
    )
    parser.add_argument(
        '--prefix-hidden-size',
        type=int, default=512
    )
    parser.add_argument(
        '--problem-type',
        type=str, default='single-label-classification',
        help='type of problem. supported: single-label-classification'
    )
    
    # Train
    parser.add_argument(
        '--batch-size',
        type=int, default=32,
        help='batch size'
    )
    parser.add_argument(
        '--lr',
        type=int, default=0.0001,
        help='learning rate'
    )
    parser.add_argument(
        '--max-epoch',
        type=int, default=100,
        help='maximum epoch'
    )
    parser.add_argument(
        '--device',
        type=str, default="cuda",
        help='set cuda if you want to use gpu, else cpu'
    )
    parser.add_argument(
        '--name', '-n',
        type=str, default="0000000",
        help='see gen text option'
    )
    
    args = parser.parse_args()

    assert args.num_graph_convs + 1 == args.prefix_len, \
        "Not yet implemented"

    if args.debug:
        args.batch_size = 8
        args.max_epoch = 1
    
    return args


if __name__ == '__main__':
    args = get_args()
    