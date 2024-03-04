import os
import sys
import argparse
import logging
import random
import shutil
import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

# from DPDN import Net
from solver import test_func, get_logger
from dataset import TestDataset
from evaluation_utils import evaluate

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="1",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default="config/psg6d_default.yaml",
                        help="path to config file")
    parser.add_argument("--dataset",
                        type=str,
                        default="REAL275",
                        help="[REAL275 | CAMERA25]")
    parser.add_argument("--data_name",
                        type=str,
                        default="real",
                        help="[real | camera]")
    parser.add_argument("--test_epoch",
                        type=int,
                        default=30,
                        help="test epoch")
    parser.add_argument('--mask_label', action='store_true', default=False,
                        help='whether having mask labels of real data')
    parser.add_argument('--only_eval', action='store_true', default=False,
                        help='whether directly evaluating the results')
    args_cfg = parser.parse_args()
    return args_cfg

def init():
    args = get_parser()
    exp_name = args.config.split("/")[-1].split(".")[0] 

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.dataset = args.dataset  #REAL275
    cfg.data_name = args.data_name #real
    if cfg.test.test_path is not None:
        log_dir = cfg.test.test_path
    else:
        log_dir = os.path.join("log", exp_name, args.dataset) 
    cfg.log_dir = log_dir
    cfg.gpus = args.gpus
    cfg.test_epoch = args.test_epoch
    cfg.mask_label = args.mask_label
    cfg.only_eval = args.only_eval

    if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    logger = get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/test_epoch" + str(cfg.test_epoch)  + "_logger.log")

    return logger, cfg

if __name__ == "__main__":
    logger, cfg = init()

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    save_path = os.path.join(cfg.log_dir, 'eval_epoch' + str(cfg.test_epoch))

    if not cfg.only_eval:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # model
        logger.info("=> creating model ...")
        if cfg.model_arch == "psg6d":
            from psg6d import PSG6D
            model = PSG6D(cfg.num_category, cfg.freeze_world_enhancer)

        if len(cfg.gpus)>1:
            model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
        model = model.cuda()

        checkpoint = os.path.join(cfg.log_dir, 'epoch_' + str(cfg.test_epoch) + '.pth')
        if not os.path.exists(checkpoint):
            shutil.copy(os.path.join('log', cfg. exp_name, 'epoch_30.pth'), cfg.log_dir)

        logger.info("=> loading checkpoint from path: {} ...".format(checkpoint))
        gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

        # data loader

        TestingDataset = TestDataset
        dataset = TestingDataset(cfg.test, BASE_DIR, cfg.dataset)
        dataloder = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=8,
                shuffle=False,
                drop_last=True
            )
        test_func(model, dataloder, save_path, cfg.exp_name, cfg.dataset, cfg.data_name)

    evaluate(save_path, logger)

