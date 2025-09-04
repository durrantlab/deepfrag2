"""DeepFrag model combined with ESM-2 embeddings."""

import os
import sys
import torch
import logging
import argparse
from torch import nn
from torch import hub
from typing import List, Optional
from apps.deepfrag.model import DeepFragModel
from collagen.external.common.types import StructureEntry
from collagen.core.molecules.fingerprints import download_molbert_ckpt, _molbert

try:
    import esm
except:
    print("Library esm is not installed...")

ON_GPU = torch.cuda.is_available()
ESM2_MODEL = None
BATCH_CONVERTER = None


def download_esm2_model(esm2_model_name):
    global ESM2_MODEL
    global BATCH_CONVERTER

    # set the directory where the ESM-2 model will be downloaded
    hub.set_dir(os.getcwd() + os.sep + "esm2" + os.sep + esm2_model_name)

    # download and load the ESM-2 model
    ESM2_MODEL, alphabet = esm.pretrained.load_model_and_alphabet_hub(esm2_model_name)
    BATCH_CONVERTER = alphabet.get_batch_converter()
    ESM2_MODEL.eval()  # disables dropout for deterministic results
    if ON_GPU:
        ESM2_MODEL = ESM2_MODEL.cuda()
        print("ESM-2 is using cuda: " + esm2_model_name)
    else:
        print("ESM-2 is using cpu: " + esm2_model_name)


class DeepFragModelESM2(DeepFragModel):
    """DeepFrag model combined with ESM-2 embeddings."""

    def __init__(self, **kwargs):
        """Initialize the model.

        Args:
            **kwargs: additional keyword arguments.
        """
        super().__init__(**kwargs)

        if not kwargs["fusing_strategy_for_mm"]:
            raise Exception("It must be specified a strategy to combine modalities.")
        if kwargs["fusing_strategy_for_mm"] != "concatenate":
            raise Exception("The fusing strategy is only concatenate in multi-modal learning .")

        self.esm2_model_for_mm = False
        self.molbert_model_for_mm = False
        self.combined_embedding_size = 512 if kwargs["fusing_strategy_for_mm"] == "concatenate" else 0

        for esm2_model_name in ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D",
                                "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"]:
            if kwargs["esm2_model_for_mm"] == esm2_model_name:
                # download and load the ESM-2 model
                download_esm2_model(esm2_model_name)
                self.num_layers = ESM2_MODEL.num_layers

                # hashtable(seq, embedding): to avoid repeated calculation of embeddings
                self.embedding_per_seq = {}

                # attributes to work with the combined multimodal features
                self.combined_embedding_size += DeepFragModelESM2.__get_embedding_size(kwargs["esm2_model_for_mm"])
                self.esm2_model_for_mm = True
                break

        if bool(kwargs["molbert_model_for_mm"]):
            download_molbert_ckpt()
            self.embedding_per_parent = {}
            self.molbert_model_for_mm = True
            self.combined_embedding_size += 1536  # MolBert embedding size

        if not self.esm2_model_for_mm and not self.molbert_model_for_mm:
            raise Exception("ESM-2 model and/or MolBert model should be specified for multi-modal learning.")
        if not self.esm2_model_for_mm and kwargs["esm2_model_for_mm"] is not None:
            raise Exception("It should be specified a valid name for ESM-2 models.")

        self.unique_smiles = set()
        self.save_unique_smiles = None
        self.unique_sequences = set()
        self.save_unique_sequences = None
        if bool(kwargs["save_unique_smiles_and_sequences"]):
            DeepFragModelESM2.__setup_logger('unique_smiles',
                                             kwargs["default_root_dir"] + os.sep + "unique_smiles.log")
            DeepFragModelESM2.__setup_logger('unique_sequences',
                                             kwargs["default_root_dir"] + os.sep + "unique_sequences.log")
            self.save_unique_smiles = logging.getLogger('unique_smiles')
            self.save_unique_smiles.propagate = False
            self.save_unique_sequences = logging.getLogger('unique_sequences')
            self.save_unique_sequences.propagate = False

        self.reduction_combined_embedding = nn.Sequential(
            # Randomly zero some values
            nn.Dropout(),
            # Linear transform.
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            nn.Linear(self.combined_embedding_size, 512),
            # Activation function. Output 0 if negative, same if positive.
            nn.ReLU(),
        )

    @staticmethod
    def __setup_logger(logger_name, log_file, level=logging.INFO):
        log_setup = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        log_setup.setLevel(level)
        log_setup.addHandler(file_handler)
        # log_setup.addHandler(stream_handler)

    @staticmethod
    def __get_embedding_size(esm2_model_name):
        if esm2_model_name == "esm2_t6_8M_UR50D":
            return 320
        if esm2_model_name == "esm2_t12_35M_UR50D":
            return 480
        if esm2_model_name == "esm2_t30_150M_UR50D":
            return 640
        if esm2_model_name == "esm2_t33_650M_UR50D":
            return 1280
        if esm2_model_name == "esm2_t36_3B_UR50D":
            return 2560

        return 5120  # corresponds to esm2_t48_15B_UR50D

    @staticmethod
    def add_model_args(
            parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add model-specific arguments to the parser.

        Args:
            parent_parser (argparse.ArgumentParser): The parser to add to.

        Returns:
            argparse.ArgumentParser: The parser with model-specific arguments added.
        """
        # For many of these, good to define default values in args_defaults.py
        parser = parent_parser.add_argument_group("DeepFragModelESM2")
        parser.add_argument(
            "--run_mm_model",
            action="store_true",
            help="If given, it will be applied a multi-model strategy.",
        )
        parser.add_argument(
            "--esm2_model_for_mm",
            required=False,
            type=str,
            help="The ESM-2 model to be used to compute evolutionary embeddings to be used in the multi-modal learning:\n"
                 "esm2_t6_8M_UR50D\n"
                 "esm2_t12_35M_UR50D\n"
                 "esm2_t30_150M_UR50D\n"
                 "esm2_t33_650M_UR50D\n"
                 "esm2_t36_3B_UR50D\n"
                 "esm2_t48_15B_UR50D\n",
        )
        parser.add_argument(
            "--molbert_model_for_mm",
            action="store_true",
            help="If given, the MolBERT model will be used in the multi-modal learning.",
        )
        parser.add_argument(
            "--fusing_strategy_for_mm",
            required=False,
            type=str,
            help="The strategy to used in the multi-modal learning to combine the output of the different modalities:\n"
                 "concatenate\n"
                 "average\n",
        )
        parser.add_argument(
            "--save_unique_smiles_and_sequences",
            action="store_true",
            help="If given, the smiles representing ligands and amino acid sequences representing receptors are saved.",
        )
        return parent_parser

    def forward(self, voxel: torch.Tensor, entry_infos: Optional[List[StructureEntry]] = None) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            voxel (torch.Tensor): The voxel grid.
            entry_infos: the information for each voxel

        Returns:
            torch.Tensor: The predicted fragment fingerprint.
        """
        latent_space = self.encoder(voxel)
        if ON_GPU is True and latent_space.get_device() == -1:
            latent_space = latent_space.cuda()

        try:
            if entry_infos is not None:
                combined_latent_space = torch.zeros((latent_space.size()[0], self.combined_embedding_size), dtype=torch.float32)
                if ON_GPU is True and combined_latent_space.get_device() == -1:
                    combined_latent_space = combined_latent_space.cuda()

                for idx, entry_info in enumerate(entry_infos):
                    latent_space_idx = latent_space[idx]
                    if self.esm2_model_for_mm:
                        # concatenate with ESM-2 embedding
                        latent_space_idx = torch.cat((latent_space_idx, self.__esm2_model_processing(entry_info)))

                    if self.molbert_model_for_mm:
                        # concatenate with MolBert embedding
                        latent_space_idx = torch.cat((latent_space_idx, self.__molbert_model_processing(entry_info)))

                    combined_latent_space[idx] = latent_space_idx

                    if self.save_unique_sequences and entry_info.receptor_sequence not in self.unique_sequences:
                        self.unique_sequences.add(entry_info.receptor_sequence)
                        self.save_unique_sequences.info(entry_info.receptor_sequence)

                    if self.save_unique_smiles and entry_info.parent_smiles not in self.unique_smiles:
                        self.unique_smiles.add(entry_info.parent_smiles)
                        self.save_unique_smiles.info(entry_info.parent_smiles)

                # apply linear layers to decrease the combined feature tensor to dimension 512
                latent_space = self.reduction_combined_embedding(combined_latent_space)
        except Exception as e:
            print("Sequence error: ", e, file=sys.stderr)

        fps = self.deepfrag_after_encoder(latent_space)
        return fps

    def __esm2_model_processing(self, entry_info):
        hash_id = hash(entry_info.receptor_sequence)
        if hash_id not in self.embedding_per_seq.keys():
            # building the input data to the ESM-2 model
            batch_labels, batch_strs, batch_tokens = BATCH_CONVERTER([(hash_id, entry_info.receptor_sequence)])
            if ON_GPU is True and batch_tokens.get_device() == -1:
                batch_tokens = batch_tokens.cuda()

            # Extract per-residue representations
            with torch.no_grad():
                result = ESM2_MODEL(batch_tokens, repr_layers=[self.num_layers], return_contacts=False)
                token_representation = result["representations"][self.num_layers]
                esm2_embedding = token_representation[0, 1:len(batch_strs[0]) + 1].mean(0)
                self.embedding_per_seq[hash_id] = esm2_embedding
        else:
            esm2_embedding = self.embedding_per_seq[hash_id]

        return esm2_embedding

    def __molbert_model_processing(self, entry_info):
        hash_id = hash(entry_info.parent_smiles)
        if hash_id not in self.embedding_per_parent.keys():
            molbert_embedding = torch.tensor(_molbert(m=None, size=0, smiles=entry_info.parent_smiles))
            if ON_GPU is True and molbert_embedding.get_device() == -1:
                molbert_embedding = molbert_embedding.cuda()
            self.embedding_per_parent[hash_id] = molbert_embedding
        else:
            molbert_embedding = self.embedding_per_parent[hash_id]

        return molbert_embedding
