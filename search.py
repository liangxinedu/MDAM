import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model_search, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
from nets.attention_model import set_decode_type


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model_search(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda

    device = torch.device("cuda:0" if use_cuda else "cpu")
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
    results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)
    return

def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)

    model.set_decode_type("greedy")

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    import datetime
    a = datetime.datetime.now()
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            set_decode_type(model, "greedy")
            model.train()
            _, costs = model(batch, beam_size=opts.beam_size, fst=1)
            results.append(costs)

    results = torch.cat(results, 0)

    b = datetime.datetime.now()
    delta = b - a

    print (results.mean().item(), delta)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")


    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")


    parser.add_argument('--beam_size', type=int, help="beam size")

    opts = parser.parse_args()




    widths = opts.width if opts.width is not None else [0]

    for width in widths:
        for dataset_path in opts.datasets:
            eval_dataset(dataset_path, width, opts.softmax_temperature, opts)
