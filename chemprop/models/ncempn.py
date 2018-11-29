from argparse import Namespace
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import create_mask, index_select_ND, visualize_atom_attention, visualize_bond_attention, \
    get_activation_function
from chemprop.models.mpn import MPNEncoder

class NCEMPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        super(NCEMPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or self.atom_fdim + get_bond_fdim(args)
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)
        self.next_encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)
        self.neg_encodings = None

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                next_batch: Union[List[str], BatchMolGraph],
                neg_batch: Union[List[str], BatchMolGraph]) -> torch.Tensor:

        batch = mol2graph(batch, self.args)
        next_batch = mol2graph(next_batch, self.args)
        neg_batch = mol2graph(neg_batch, self.args)  # TODO in the future could recreate neg at different intervals

        output = self.encoder.forward(batch, None)
        next_output = self.next_encoder.forward(next_batch, None)
        neg_output = self.next_encoder.forward(neg_batch, None)

        return (output, next_output, neg_output)
    
    # def recreate_neg_encodings(self, neg_batch):
    #     neg_batch = mol2graph(neg_batch, self.args)
    #     self.neg_encodings = self.next_encoder.forward(neg_batch)