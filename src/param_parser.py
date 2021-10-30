import argparse


def parameter_parser():

    parser = argparse.ArgumentParser()

    """ 
    hyperparameters
    """

    parser.add_argument("--epochs",
                        type=int,
                        default=5,
                        help="Number of training epochs. Default is 5.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
                        help="Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
                        help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
                        help="Similarity score bins. Default is 16.")

    parser.add_argument("--perspectives",
                        type=int,
                        default=50,
                        help='number of perspectives for matching')

    parser.add_argument("--hidden-size",
                        type=int,
                        default=20,
                        help='hidden size ')

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--node-nhid-1",
                        default=64)
    parser.add_argument("--node-nhid-2",
                        default=32)
    parser.add_argument("--node-nhid-3",
                        default=16)

    parser.add_argument("--edge-nhid-1",
                        default=64)
    parser.add_argument("--edge-nhid-2",
                        default=32)
    parser.add_argument("--edge-nhid-3",
                        default=16)

    parser.add_argument("--gcn-out",
                        default=16)
    parser.add_argument("--embedding-out",
                        default=32)

    parser.add_argument("--label_cut_ratio",
                        default=1)

    """
    model settings
    """

    parser.add_argument("--attention-module",
                        default=True)

    parser.add_argument("--tensor-network",
                        default=True)

    parser.add_argument("--histogram",
                        default=True)

    parser.add_argument("--node-graph-matching",
                        default=True)

    """
    experiment settings
    """

    parser.add_argument("--small-dataset",
                        default=False)

    parser.add_argument("--save-model",
                        default=True)

    parser.add_argument("--save-path",
                        type=str,
                        default=None,
                        help="Where to save the trained model")

    parser.add_argument("--load-model",
                        default=False)

    parser.add_argument("--load-path",
                        type=str,
                        default=None,
                        help="Load a pretrained model")

    parser.add_argument("--validate",
                        default=True)

    """ 
    dataset settings
    """

    parser.add_argument("--current-dataset-name",
                        default="LINUX")

    parser.add_argument("--half-dataset",
                        default=False)

    parser.add_argument("--dataset-root-path",
                        default="../datasets/")

    """ 
    commandline settings
    """

    parser.add_argument("--device_num",
                        default='0')
    parser.add_argument("--filename",
                        default="temp")

    return parser.parse_args()
