from utils import tab_printer
from graph_sim import GraphSimTrainer
from param_parser import parameter_parser
import torch
import os


def main():
    args = parameter_parser()

    device_num = args.device_num
    assert device_num == '0' or device_num == '1'

    args.device = torch.device(('cuda:' + device_num) if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = device_num

    tab_printer(args)

    trainer = GraphSimTrainer(args)
    if args.load_model:
        trainer.load()
    else:
        trainer.train()

    if args.save_model:
        trainer.save()

    trainer.test()


if __name__ == "__main__":
    main()
