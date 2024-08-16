import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MCL")
    # parser.add_argument("--trainFile", default="train.txt", help="")

    parser.add_argument("--seed", type=int, default=2023, help="random seed for init")
    parser.add_argument("--model", default="TAHIN", help="Model Name")
    parser.add_argument(
        "--dataset",
        default="Movielens",
        help="Dataset to use, default: Movielens",
    )
    parser.add_argument("--multicore", type=int, default=0, help="use multiprocessing or not in test")
    parser.add_argument("--data_path", nargs="?", default="./data/", help="Input data path.")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout", type=bool, default=False, help="consider node dropout or not")
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]', help='Output sizes of every layer')
    parser.add_argument('--test_batch_size', type=int, default=100, help='batch size')
    parser.add_argument("--mess_keep_prob", nargs='?', default='[0.1, 0.1, 0.1]', help="ratio of node dropout")
    parser.add_argument("--node_keep_prob", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--dim', type=int, default=128, help='embedding size')

    parser.add_argument('--drop_ratio', type=float, default=0.5, help='l2 regularization weight')

    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')

    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of processes to construct batches",
    )

    parser.add_argument(
        "--in_size",
        default=128,
        type=int,
        help="Initial dimension size for entities.",
    )
    parser.add_argument(
        "--out_size",
        default=128,
        type=int,
        help="Output dimension size for entities.",
    )

    parser.add_argument(
        "--num_heads", default=1, type=int, help="Number of attention heads"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default="0",
        help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")

    parser.add_argument('--topK', nargs='?', default='[10]', help='size of Top-K')

    parser.add_argument("--verbose", type=int, default=10, help="Test interval")

    parser.add_argument('--regs', nargs='?', default='[1e-4]',
                        help='Regularizations.')

    parser.add_argument('--GCNLayer', type=int, default=3, help="the layer number of GCN")

    # Contrast learing
    parser.add_argument(
        "--cl_rate", default=0.01, type=float, help="the proportion of cl_loss."
    )
    parser.add_argument("--temperature", default=0.6, type=float, help=".")
    parser.add_argument("--lam", default=0.5, type=float, help=".")
    parser.add_argument("--cl_hidden_dim", default=128, type=int, help=".")
    parser.add_argument("--pos_num", default=20, type=int, help="筛选正样本的门槛值.")

    # IB
    parser.add_argument("--IB_rate", default=0.005, type=float, help=".")
    parser.add_argument("--mask_reg", default=0.01, type=float, help=".")
    parser.add_argument("--choosing_tmp", default=0.2, type=float, help=".")
    parser.add_argument("--ssl_temp", default=0.2, type=float, help=".")
    parser.add_argument("--walk_length", default=10, type=int, help=".")
    parser.add_argument("--LCN_layer", default=2, type=int, help=".")
    parser.add_argument("--sigma", default=2, type=float, help=".")

    # align and uniform
    parser.add_argument("--gamma", default=1.0, type=float, help="self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2.")
    parser.add_argument("--beta", default=0.01, type=float, help=".")
    return parser.parse_args()