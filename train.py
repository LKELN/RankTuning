import math
import torch
import logging
import numpy as np
from tqdm import tqdm, trange
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import loss_sum

torch.backends.cudnn.benchmark = True  # Provides a speedup

import util
import test
import parser
import commons
import datasets_ws
import network
from loss import GeneralizedRecallloss
import warnings
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")
import os


args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
args.runpath = "./runs"
if not os.path.exists(args.runpath):
    os.makedirs(args.runpath)
    print("runspath is not exist,making it")
writer = SummaryWriter(logdir=join(args.runpath, start_time.strftime('%Y-%m-%d_%H-%M-%S')))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading dataset {args.eval_dataset_name} from folder {args.eval_datasets_folder}")

triplets_ds = datasets_ws.TripletsDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "train", args.negs_num_per_query)
logging.info(f"Train query set: {triplets_ds}")
val_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
logging.info(f"Test set: {test_ds}")

#### Initialize model
model = network.GeoLocalizationNet(args)

model = model.to(args.device)
Global_ranktuning = torch.nn.DataParallel(model.Global_ranktuning)
model = torch.nn.DataParallel(model)

## Freeze parameters except adapter
for i, (name, param) in enumerate(model.module.backbone.named_parameters()):
    if i < 117:
        param.requires_grad = False
for i, (name, param) in enumerate(model.named_parameters()):
    if "Global_ranktuning" not in name:
        param.requires_grad = False
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=0.001)
elif args.optim == "adamw":
    optimizer = torch.optim.AdamW(parameters, weight_decay=9.5e-9, lr=args.lr)


#### Resume model, optimizer, and other training parameters
if args.resume:
    model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
    best_r5 = start_epoch_num = not_improved_num = 0

else:
    best_r5 = start_epoch_num = not_improved_num = 0

logging.info(f"Output dimension of the model is {args.features_dim}")

#### Getting GSVCities
Ranktuning_loss = GeneralizedRecallloss(k_vals=[1,2,4], tmp1=0.01, tmp2=0.1,tmp3=0.1).cuda()

for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")

    epoch_start_time = datetime.now()
    epoch_sort_loss = np.zeros((0, 1), dtype=np.float16)
    model = model.train()
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        # break
        logging.debug(f"Cache: {loop_num} / {loops_num}")

        # Compute triplets to use in the triplet loss
        if (args.cache_refresh_rate != args.queries_per_epoch) or epoch_num == 0:
            triplets_ds.is_inference = True
            triplets_ds.compute_triplets(args, model)
            triplets_ds.is_inference = False

        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size, collate_fn=datasets_ws.collate_fn,
                                 pin_memory=(args.device == "cuda"), drop_last=True)
        if args.fix:
            model = model.eval()
        else:
            model = model.train()

        Global_ranktuning = Global_ranktuning.train()
        for images, triplets_local_indexes, triplets_global_indexes, utms in tqdm(triplets_dl, ncols=100):
            # Flip all triplets or none
            if args.horizontal_flip:
                images = transforms.RandomHorizontalFlip()(images)

            # Compute features of all images (images contains queries, positives and negatives)
            global_features = model(images.to(args.device))

            sort_values = Global_ranktuning(global_features,args)

            loss = Ranktuning_loss(sort_values, args, triplets_global_indexes)
            descriptors = nn.functional.normalize(global_features.flatten(1), dim=-1, p=2)

            del descriptors, sort_values, global_features

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Keep track of all losses by appending them to epoch_losses
            batch_sort_loss = loss.item()
            epoch_sort_loss = np.append(epoch_sort_loss, batch_sort_loss)
            del loss

        writer.add_scalar('Average_epoch_sort_loss', epoch_sort_loss.mean(),
                          epoch_num*loops_num+loop_num)
        writer.add_scalar('Loss', epoch_sort_loss.mean(),
                          epoch_num*loops_num+loop_num)

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average sort epoch loss= {epoch_sort_loss.mean():.4f}")
    writer.add_scalar('Total_loss', epoch_sort_loss.mean(),epoch_num)
    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

    is_best = recalls[0] > best_r5

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r5": best_r5,
                                "not_improved_num": not_improved_num
                                }, is_best, filename="last_model.pth")

    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@1 = {best_r5:.1f}, current R@1 = {(recalls[0] ):.1f}")
        best_r5 = recalls[0]
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: best R@1 = {best_r5:.1f}, current R@1 = {(recalls[0] ):.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

logging.info(f"Best R@1: {best_r5:.1f}")
logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
logging.info("Test *best* model on test set")
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)
recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

#### Test last model on test set
logging.info("Test *last* model on test set")
last_model_state_dict = torch.load(join(args.save_dir, "last_model.pth"))["model_state_dict"]
model.load_state_dict(last_model_state_dict)
recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")



