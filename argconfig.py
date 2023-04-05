import argparse
import ruamel.yaml as yaml
from fvcore.common.config import CfgNode as CN
import os

def default_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default=None,
        type=str,
        help="/path/to/config-file"
    )
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def load():
    args = default_args().parse_args()
    print(args)
    print(args.config_file)
    if args.config_file is not None:
        cfg = CN((yaml.safe_load(open(args.config_file, 'r'))))
    else:
        os.error('Please specify config-file')
    cfg.merge_from_list(args.opts)
    
    return prep(cfg)

def prep(cfg, print_cfg=True):
    cfg.cfg_name = '{}_wd{}_bs{}_epoch{}_lr{}_{}_lamda_{:0.1f}_bit{}_{}{}'.format(
                                                    cfg.optim.loss_type,
                                                    cfg.optim.enable_decay,
                                                    cfg.optim.batch_size, 
                                                    cfg.optim.epochs, 
                                                    cfg.optim.lr_core, 
                                                    cfg.optim.lr_mask,
                                                    cfg.optim.lamda_ini,
                                                    cfg.model.wgt_bit, 
                                                    cfg.model.act_bit,
                                                    cfg.misc.suffix                                                 
                                                    )
    cfg.file_name = "{:08d}".format(int(cfg.optim.spr_w))
    # create dir for score
    cfg.root_dir  = os.path.join(cfg.misc.log_dir, cfg.dataset.name, cfg.model.name, cfg.cfg_name, cfg.file_name)
    cfg.stats_dir = os.path.join(cfg.root_dir, 'score')
    cfg.model_dir = os.path.join(cfg.root_dir, 'model')    
    
    os.makedirs(cfg.stats_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    cfg.model_path_final = os.path.join(cfg.model_dir, 'final.pt')
    cfg.model_path_best = os.path.join(cfg.model_dir, 'best.pt')
    with open(os.path.join(cfg.root_dir, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    if print_cfg:
        print(cfg)
    return cfg