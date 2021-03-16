
import argparse
import json
import logging
import os
import pathlib
import resource
from typing import List

import numpy as np
from openbabel import pybel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml

from ..molecule_util import MolGraph
from ..models.model_interfaces import MolAutoencoder
from ..models.dense_graph_autoencoder import DenseGraphAutoencoder
from ..data_formats.zinc import ZINCMolGraphProviderH5


def disable_ob_logging():
    pybel.ob.obErrorLog.SetOutputLevel(0)


def mse(a,b):
    return torch.sum((a-b)**2)


def do_batch(mod: MolAutoencoder, loss_fn, batch: List[MolGraph], mask_ratio: float):
    acc = 0
    atom_acc = 0
    loss = 0
    
    for b in batch:
        orig_types = b.atom_types
        
        z = mod.encode(b)
        
        z_keep = torch.zeros_like(z)
        
        # Randomly zero out some entries.
        for i in range(len(z)):
            if np.random.rand() > mask_ratio:
                z_keep[i] = z[i]
        
        p = mod.decode(z_keep, b)
        
        loss += loss_fn(p.atom_types, orig_types)
        
        comp = torch.round(p.atom_types) == orig_types
        acc += torch.mean(comp.float())
        
        # Where is the prediction completely correct for an atom?
        perfect_atoms = torch.all(comp, axis=1)
        atom_acc += torch.mean(perfect_atoms.float())
        
    acc /= len(batch)
    atom_acc /= len(batch)
    loss /= len(batch)
    
    return {
        'acc': acc,
        'atom_acc': atom_acc,
        'loss': loss
    }


def train(attr: dict, save_path: str, wandb_project: str, cpu_only: bool):

    # Necessary to prevent a multi-worker DataLoader from crashing.
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    device = torch.device('cpu') if cpu_only else torch.device('cuda')

    # Save a copy of the configuration file.
    logging.info('Saving training configuration...')
    os.makedirs(save_path, exist_ok=True)
    p = pathlib.Path(save_path)
    with open(p / 'train_config.yaml', 'w') as f:
        f.write(yaml.dump(attr))

    # Load ZINC training data.
    logging.info('Loading dataset...')
    dat = ZINCMolGraphProviderH5(attr['zinc_h5'], make_3D=False)

    loader = DataLoader(
        dat, batch_size=attr['batch_size'], shuffle=True, 
        collate_fn=lambda x:x)

    # Initialize the model.
    logging.info('Creating model...')
    mod = DenseGraphAutoencoder.create(attr['model_attr'])
    mod.models['encoder'].to(device)
    mod.models['decoder'].to(device)

    opt = torch.optim.Adam(
        list(mod.models['encoder'].parameters()) + list(mod.models['decoder'].parameters()),
        lr=float(attr['lr'])
    )

    if wandb_project is not None:
        wandb.init(
            project=wandb_project,
            reinit=True,
            config=attr
        )

    logging.info('Training!')   
    for epoch in range(attr['num_epochs']):
        logging.info(f'Starting epoch {epoch}')

        num_steps = min(len(loader), attr['max_steps_per_epoch'])
        curr_step = 0
        for idx, batch in tqdm(enumerate(loader), desc=f'Epoch {epoch}', total=num_steps):
            curr_step += 1
            if curr_step >= num_steps:
                # Exit early once we reach max_steps_per_epoch.
                break

            for x in batch:
                x.to(device)

            res = do_batch(mod, mse, batch, attr['mask_ratio'])
            
            opt.zero_grad()
            res['loss'].backward()
            opt.step()
            
            if wandb_project is not None:
                wandb.log({
                    'acc': res['acc'],
                    'atom_acc': res['atom_acc'],
                    'loss': res['loss'].detach()
                })

        logging.info(f'Saving model...')
        mod.save(str(pathlib.Path(save_path) / 'model'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--save_path', required=True, help='Path to target run directory.')
    parser.add_argument('--wandb', required=False, default=None,
        help='Name of wandb project (optional).')
    parser.add_argument('--cpu', default=False, action='store_true',
        help='Disable CUDA GPU acceleration.')
    args = parser.parse_args()

    attr = yaml.safe_load(open(args.config, 'r').read())

    disable_ob_logging()

    train(
        attr, 
        save_path=args.save_path, 
        wandb_project=args.wandb, 
        cpu_only=args.cpu
    )


if __name__=='__main__':
    main()
