

from typing import TYPE_CHECKING, List, Optional, Callable, Any, Tuple, Union

from torch.utils.data import Dataset
import numpy as np
from collagen.external.moad.split import full_moad_split

if TYPE_CHECKING:
    from collagen.external.moad.interface import MOADInterface
    from collagen.core.molecules.mol import Mol
    from collagen.external.moad.types import MOAD_split
    from collagen.external.moad.types import MOAD_target

# TODO: NOT CURRENTLY USED

def _unit_rand(thresh):
    u = np.random.uniform(size=3)
    u = (u * 2) - 1
    u /= np.sqrt(np.sum(u * u))
    u *= np.random.rand() * np.sqrt(thresh)
    return u


def _sample_near(coords, thresh):
    idx = np.random.choice(len(coords))
    c = coords[idx]

    offset = _unit_rand(thresh)
    p = c + offset

    return p


def _sample_inside(bmin, bmax, thresh, avoid):
    while True:
        p = np.random.uniform(size=3)
        p = (p * (bmax - bmin)) + bmin

        bad = False
        for i in range(len(avoid)):
            d = np.sum((avoid[i] - p) ** 2)
            if np.sqrt(d) <= thresh + 1e-5:
                bad = True
                break

        if bad:
            continue

        return p


class MOADPocketDataset(Dataset):

    """
    A Dataset that provides (receptor, pos, neg) tuples where pos and neg are points in a binding pocket and outside of a binding pocket respectively.

    Positive samples are genearated by picking a random ligand atom and sampling a random offset. Negative samples are generated
    by randomly sampling a point withing the bounding box of the receptor (plus padding) that is not near any ligand atom.

    Args:
        moad (MOADInterface): An initialized MOADInterface object.
        thresh (float, optional): Threshold to ligand atoms to consider a "binding pocket."
        padding (float, optional): Padding added to receptor bounding box to sample negative examples.
        split (MOAD_split, optional): An optional split to constrain the space of examples.
        transform (Callable[[Mol, np.ndarray, np.ndarray], Any], optional): An optional transformation function to invoke before returning samples.
            Takes the arguments (receptor, pos, neg).
    """

    def __init__(
        self,
        moad: "MOADInterface",
        thresh: float = 3,
        padding: float = 5,
        split: Optional["MOAD_split"] = None,
        transform: Optional[
            Callable[["Mol", "np.ndarray", "np.ndarray"], Any]
        ] = None,
        **kwargs
    ):
        self.moad = moad
        self.thresh = thresh
        self.padding = padding
        self.split = split if split is not None else full_moad_split(moad)
        self.transform = transform
        self._index = self._build_index()

    def _build_index(self):
        index: List[Tuple[str, int]] = []
        for t in sorted(self.split.targets):
            for n in range(len(self.moad[t])):
                index.append((t, n))
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Union[None, Tuple["Mol", "np.ndarray", "np.ndarray"]]:
        target, n = self._index[idx]

        try:
            moad_target: "MOAD_target" = self.moad[target]
            # TODO: jacob version
            # rec, ligs = moad_target[n]
            # TODO: cesar version. Why the change?
            rec, ligs = self.moad[target][n]
        except:
            return None

        if len(ligs) == 0:
            return None

        lig_coords = np.concatenate([x.coords for x in ligs])
        rec_coords = rec.coords

        pos = _sample_near(lig_coords, self.thresh)

        box_min = np.min(rec_coords, axis=0) - self.padding
        box_max = np.max(rec_coords, axis=0) + self.padding

        neg = _sample_inside(box_min, box_max, self.thresh, avoid=lig_coords)

        out = (rec, pos, neg)

        if self.transform is not None:
            out = self.transform(out)

        return out
