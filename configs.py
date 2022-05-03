import argparse


def get_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cuda', action='store_true',
    #                     help="use GPU(s) to speed up training")
    parser.add_argument('-n', default=75, type=int,
                        help="number of epochs to run")
    parser.add_argument('--z_dim', default=45, type=int,
                        help="size of the tensor representing the latent variable z "
                             "variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-lr', default=1e-4, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('--anneal_rate', default=0.00003, type=float,
                        help="Anneal learning rate by this factor every epoch")
    parser.add_argument('-bs', default=256, type=int,
                        help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--data_dir', type=str, default='./',
                        help='Data path')
    parser.add_argument('--l1_reg', type=float, default=0.2, help='L1 regularization coeff for gating')
    parser.add_argument('--gate_type', default='learnable', choices=['learnable, fixed'])
    parser.add_argument('--gate_subtype', default='inferred', choices=['one-one, inferred'])
    parser.add_argument('--do_train', default=False, type=bool)
    parser.add_argument('--do_test', default=True, type=bool)
    args = parser.parse_args()
    return args