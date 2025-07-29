# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
#
# Further modified by Osman Ülger (o.ulger@uva.nl) in 2025
# Changed forward pass to use automatically generated vocabulary, rather than manually specified one.
# --------------------------------------------------------

import os
import sys
import time
import datetime
import torch

from torch.utils.data import DataLoader

from X_Decoder.utils.arguments import load_opt_command

def main(args=None):
    '''
    [Main function for the entry point]
    1. Set environment variables for distributed training.
    2. Load the config file and set up the trainer.
    '''

    opt, cmdline_args = load_opt_command(args)

    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir

    # update_opt(opt, command)
    world_size = 1
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

    if opt['TRAINER'] == 'xdecoder':
        from X_Decoder.trainer import XDecoder_Trainer as Trainer
    else:
        assert False, "The trainer type: {} is not defined!".format(opt['TRAINER'])

    trainer = Trainer(opt)
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    trainer.eval()


if __name__ == "__main__":
    main()
    sys.exit(0)