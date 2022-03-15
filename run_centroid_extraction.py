#############################################################################################################
# Run simplex extraction on novel dataset
#############################################################################################################

import os
from tqdm import tqdm 
import torch
import numpy as np
import random
import warnings
import pickle
from src.args import process_arguments
from src.utils import fix_seed, load_features
from src.solver import gradient_descent_ClosedForm, find_summetsBatch, find_summetsKmeans
warnings.filterwarnings("ignore")

args = process_arguments()
fix_seed(args.seed, deterministic=args.deterministic)



print(f'{args.n_ways}-ways | {args.n_shots[0]}-shots | total runs: {args.n_runs} | AS: {args.AS} | QR:{not args.notQR} | \u03BB:{args.lamda_reg} | seed: {args.seed} | simplex: {args.simplex} | alpha-iter: {args.alpha_iter} | threshold-elbow: {args.thresh_elbow} | Preprocessing : {args.preprocessing} | Postprocessing : {args.postprocessing}')

novel_features, _ = load_features(args.features_path)
print(novel_features.shape)
novel_features = novel_features.reshape(-1, novel_features.shape[-2], novel_features.shape[-1])

list_D = []
for b in tqdm(range(0, novel_features.shape[0]//args.batch_size)):
    data = novel_features[b*args.batch_size:(b+1)*args.batch_size]
    if args.extraction=='simplex':
        D = find_summetsBatch(data, args, thresh_elbow=1.5, return_jumpsMSE=True, lamda_reg=args.lamda_reg, n_iter=150, alpha_iter=5, 
                                                        trainCfg={'lr':0.1, 'mmt':0.8, 'D_iter':1, 'loss_amp':10000,'loss_alpha':1}, verbose=False, maxK=4, concat=False)
    elif args.extraction=='kmeans':
        D = find_summetsKmeans(data, args, maxK=4)
    else:
        raise f'extraction options are either "simplex" or "kmeans"'
    list_D = list_D + D

if args.file_save:
    with open(args.simplex_file, 'wb') as handle:
        pickle.dump(list_D, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done :)')