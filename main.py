import torch
import numpy as np
from trainer import Trainer
import sys
from utils import *
import argparse
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument('--epoch', default = 50, type = int)
parser.add_argument('--lr', default = 0.01, type = int)
parser.add_argument('--max_size', default = 500, type = int)
parser.add_argument('--total_cls', default = 100, type = int)
args = parser.parse_args()


if __name__ == "__main__":
    t1 = time.time()
    showGod()
    trainer = Trainer(args.total_cls)
    trainer.train(args.batch_size, args.epoch, args.lr, args.max_size)
    t2 = time.time()
    print("训练时间为：", t2 - t1)
    allocated_memory = torch.cuda.memory_allocated()
    print(allocated_memory)
    print(allocated_memory / (1024 * 1024))
