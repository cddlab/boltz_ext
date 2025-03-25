from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
import math


def length(v: np.ndarray, eps: float = 1e-6) -> float:
    """Calculate the length of a vector."""
    return math.sqrt(max(eps, v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def unit_vec(v: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """Calculate the unit vector."""
    vl = length(v, eps=eps)
    return v / vl, vl


def calc_chiral_vol(crds: np.ndarray, iatm: int, aj: list[int]) -> float:
    """Calculate the chiral volume."""
    vc = crds[iatm]
    v1 = crds[aj[0]] - vc
    v2 = crds[aj[1]] - vc
    v3 = crds[aj[2]] - vc

    vol = np.dot(v1, np.cross(v2, v3))
    return vol


@dataclass
class ChiralData:
    """Class for chiral data."""

    aid0: int
    aid1: int
    aid2: int
    aid3: int
    chiral_vol: float
    w: float = 0.1
    slack: float = 0.05
    fmax: float = -100.0

    verbose: bool = False

    def setup(self, ind: int, aid: int) -> None:
        """Set up the chiral data."""
        if aid == 0:
            self.aid0 = ind
        elif aid == 1:
            self.aid1 = ind
        elif aid == 2:  # noqa: PLR2004
            self.aid2 = ind
        elif aid == 3:  # noqa: PLR2004
            self.aid3 = ind
        else:
            msg = f"Invalid data {ind=} {aid=}"
            raise ValueError(msg)

    def is_valid(self) -> bool:
        """Check if the chiral data is valid."""
        if self.aid0 >= 0 and self.aid1 >= 0 and self.aid2 >= 0 and self.aid3 >= 0:
            return True
        if self.w > 0.0:
            return True
        return False

    def reset_indices(self) -> None:
        """Reset the indices."""
        self.aid0 = -1
        self.aid1 = -1
        self.aid2 = -1
        self.aid3 = -1

    def calc(self, crds: np.ndarray) -> float:
        """Calculate the chiral data."""
        vol = calc_chiral_vol(crds, self.aid0, [self.aid1, self.aid2, self.aid3])
        if self.chiral_vol > 0:
            thr = self.chiral_vol - self.slack
        else:
            thr = self.chiral_vol + self.slack

        delta = vol - thr

        ene = delta * delta * self.w

        return ene

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> bool:
        """Calculate the gradient."""
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        a2 = crds[self.aid2]
        a3 = crds[self.aid3]
        v1 = a1 - a0
        v2 = a2 - a0
        v3 = a3 - a0
        vol = np.dot(v1, np.cross(v2, v3))

        if self.chiral_vol > 0:
            thr = self.chiral_vol - self.slack
        else:
            thr = self.chiral_vol + self.slack

        delta = vol - thr
        dE = 2.0 * delta * self.w
        # print(f"   {dE=}")

        # eps = 1e-2
        # if thr < 0:
        #     dE = max(0, dE)
        #     if dE < eps:
        #         return False
        # else:
        #     dE = min(0, dE)
        #     if dE > -eps:
        #         return False

        f1 = np.cross(v2, v3) * dE
        f2 = np.cross(v3, v1) * dE
        f3 = np.cross(v1, v2) * dE
        fc = -f1 - f2 - f3

        n1, n1l = unit_vec(f1)
        n2, n2l = unit_vec(f2)
        n3, n3l = unit_vec(f3)
        nc, ncl = unit_vec(fc)

        if self.fmax > 0:
            if n1l > self.fmax or n2l > self.fmax or n3l > self.fmax or ncl > self.fmax:
                print(f"Force mean: {(n1l + n2l + n3l + ncl) / 4}")
            n1l = min(n1l, self.fmax)
            n2l = min(n2l, self.fmax)
            n3l = min(n3l, self.fmax)
            ncl = min(ncl, self.fmax)

            f1 = n1 * n1l
            f2 = n2 * n2l
            f3 = n3 * n3l
            fc = nc * ncl

        grad[self.aid0] += fc
        grad[self.aid1] += f1
        grad[self.aid2] += f2
        grad[self.aid3] += f3
        return True

    @staticmethod
    def get_nei_atoms(iatm: int, mol: Chem.Mol) -> list[int]:
        """Get the neighboring atoms."""
        atom = mol.GetAtomWithIdx(iatm)
        aj = []
        for b in atom.GetBonds():
            j = b.GetOtherAtom(atom).GetIdx()
            aj.append(j)
        return aj

    @staticmethod
    def calc_chiral_atoms(iatm, mol, conf):
        """Calculate the chiral atoms."""
        aj = []
        ajname = []
        atom = mol.GetAtomWithIdx(iatm)
        for b in atom.GetBonds():
            j = b.GetOtherAtom(atom).GetIdx()
            aj.append(j)
            ajname.append(mol.GetAtomWithIdx(j).GetProp("name"))

        if len(aj) > 4:
            raise ValueError(f"Invalid chiral atom neighbors {iatm=} {aj=}")

        chiral_vol = calc_chiral_vol(conf.GetPositions(), iatm, aj)
        print(f"{chiral_vol=:.2f}")

        atom_name = atom.GetProp("name")
        chiral_tag = atom.GetChiralTag()
        print(f"{iatm=} {atom_name=} {aj} {ajname} {chiral_tag=}")
        return chiral_vol, aj[0], aj[1], aj[2]

    def print(self, crds: np.ndarray) -> None:
        """Print the chiral data."""
        vol = calc_chiral_vol(crds, self.aid0, [self.aid1, self.aid2, self.aid3])
        print(  # noqa: T201
            f"C {self.aid0}-{self.aid1}-{self.aid2}-{self.aid3}:"
            f" cur {vol:.2f} ref {self.chiral_vol:.2f} dif {vol - self.chiral_vol:.2f}"
        )

    def calc_sd(self, crds: np.ndarray) -> None:
        """Calculate squared difference."""
        vol = calc_chiral_vol(crds, self.aid0, [self.aid1, self.aid2, self.aid3])

        return (vol - self.chiral_vol) ** 2
