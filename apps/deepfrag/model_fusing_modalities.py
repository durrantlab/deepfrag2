"""DeepFrag model combined with ESM-2 embeddings."""

import os
import torch
import argparse
from torch import nn
from torch import hub
from typing import List, Optional
from apps.deepfrag.model import DeepFragModel
from collagen.external.common.types import StructureEntry

try:
    import esm
except:
    print("Library esm is not installed...")

ESM2_ON_GPU = False
ESM2_MODEL = None
BATCH_CONVERTER = None


def download_esm2_model(esm2_model_name, cpu: bool):
    global ESM2_MODEL
    global ESM2_ON_GPU
    global BATCH_CONVERTER

    # set directory where the ESM-2 model will be downloaded
    hub.set_dir(os.getcwd() + os.sep + "esm2" + os.sep + esm2_model_name)

    # download and load the ESM-2 model
    ESM2_MODEL, alphabet = esm.pretrained.load_model_and_alphabet_hub(esm2_model_name)
    BATCH_CONVERTER = alphabet.get_batch_converter()
    ESM2_MODEL.eval()  # disables dropout for deterministic results
    if torch.cuda.is_available() and not cpu:
        ESM2_MODEL = ESM2_MODEL.cuda()
        ESM2_ON_GPU = True
        print("Setting CUDA device for ESM-2 model")
    else:
        print("Using CPU device for ESM-2 model")


class DeepFragModelESM2(DeepFragModel):
    """DeepFrag model combined with ESM-2 embeddings."""

    def __init__(self, **kwargs):
        """Initialize the model.

        Args:
            esm2_model: ESM-2 model to be used to compute evolutionary embeddings
            num_voxel_features (int, optional): the number of features per
                voxel. Defaults to 10.
            **kwargs: additional keyword arguments.
        """
        super().__init__(**kwargs)

        for esm2_model_name in ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D",
                                "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"]:
            if kwargs["esm2_model"] == esm2_model_name:
                # download and load the ESM-2 model
                download_esm2_model(esm2_model_name, bool(kwargs["cpu"]))
                self.num_layers = ESM2_MODEL.num_layers

                # hashtable(seq, embedding): to avoid repeated calculation of embeddings
                self.embedding_per_seq = {}

                # attributes to work with the combined multimodal features
                self.combined_embedding_size = 512 + DeepFragModelESM2.__get_embedding_size(kwargs["esm2_model"])
                self.reduction_combined_embedding = self.__get_deep_layers_to_reduce_combined_features()
                break
        else:
            raise Exception("The specified ESM-2 model is not valid.")

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

    def __get_deep_layers_to_reduce_combined_features(self):
        if self.combined_embedding_size <= 1152:  # for esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D
            return nn.Sequential(
                # Randomly zero some values
                nn.Dropout(),
                # Linear transform.
                # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                nn.Linear(self.combined_embedding_size, 512),
                # Activation function. Output 0 if negative, same if positive.
                nn.ReLU(),
            )
        elif self.combined_embedding_size == 1792:  # for esm2_t33_650M_UR50D
            return nn.Sequential(
                # Randomly zero some values
                nn.Dropout(),
                # Linear transform.
                nn.Linear(1792, 896),
                # Activation function. Output 0 if negative, same if positive.
                nn.ReLU(),
                # Linear transform.
                nn.Linear(896, 512),
                # Activation function. Output 0 if negative, same if positive.
                nn.ReLU(),
            )
        elif self.combined_embedding_size == 3072:  # for esm2_t36_3B_UR50D
            return nn.Sequential(
                # Randomly zero some values
                nn.Dropout(),
                # Linear transform.
                nn.Linear(3072, 1536),
                # Activation function. Output 0 if negative, same if positive.
                nn.ReLU(),
                # Linear transform.
                nn.Linear(1536, 768),
                # Activation function. Output 0 if negative, same if positive.
                nn.ReLU(),
                # Linear transform.
                nn.Linear(768, 512),
                # Activation function. Output 0 if negative, same if positive.
                nn.ReLU(),
            )
        elif self.combined_embedding_size == 5632:  # for esm2_t48_15B_UR50D
            return nn.Sequential(
                # Randomly zero some values
                nn.Dropout(),
                # Linear transform.
                nn.Linear(5632, 2816),
                # Activation function. Output 0 if negative, same if positive.
                nn.ReLU(),
                # Linear transform.
                nn.Linear(2816, 1408),
                # Activation function. Output 0 if negative, same if positive.
                nn.ReLU(),
                # Linear transform.
                nn.Linear(1408, 704),
                # Activation function. Output 0 if negative, same if positive.
                nn.ReLU(),
                # Linear transform.
                nn.Linear(704, 512),
                # Activation function. Output 0 if negative, same if positive.
                nn.ReLU(),
            )

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
            "--esm2_model",
            required=False,
            type=str,
            help="The ESM-2 model to be used to compute evolutionary embeddings:\n"
                 "esm2_t6_8M_UR50D\n"
                 "esm2_t12_35M_UR50D\n"
                 "esm2_t30_150M_UR50D\n"
                 "esm2_t33_650M_UR50D\n"
                 "esm2_t36_3B_UR50D\n"
                 "esm2_t48_15B_UR50D\n",
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
        if ESM2_ON_GPU is True and latent_space.get_device() == -1:
            latent_space = latent_space.cuda()

        try:
            if entry_infos is not None:
                combined_latent_space = torch.zeros((latent_space.size()[0], self.combined_embedding_size),
                                                    dtype=torch.float32)

                for idx, entry_info in enumerate(entry_infos):
                    hash_id = hash(entry_info.receptor_sequence)
                    if hash_id not in self.embedding_per_seq.keys():
                        # building the input data to the ESM-2 model
                        batch_labels, batch_strs, batch_tokens = BATCH_CONVERTER(
                            [(hash_id, entry_info.receptor_sequence)])
                        if ESM2_ON_GPU is True and batch_tokens.get_device() == -1:
                            batch_tokens = batch_tokens.cuda()

                        # Extract per-residue representations
                        with torch.no_grad():
                            result = ESM2_MODEL(batch_tokens, repr_layers=[self.num_layers], return_contacts=False)
                            token_representation = result["representations"][self.num_layers]
                            esm2_embedding = token_representation[0, 1:len(batch_strs[0]) + 1].mean(0)
                            self.embedding_per_seq[hash_id] = esm2_embedding
                    else:
                        esm2_embedding = self.embedding_per_seq[hash_id]

                    # concatenate CNN latent space with ESM-2 embedding
                    latent_space_idx = torch.cat((latent_space[idx], esm2_embedding))
                    combined_latent_space[idx] = latent_space_idx

                # apply linear layers to decrease the combined feature tensor to dimension 512
                latent_space = self.reduction_combined_embedding(combined_latent_space)
        except Exception as e:
            print("Sequence error: ", e, file=sys.stderr)

        fps = self.deepfrag_after_encoder(latent_space)
        return fps
