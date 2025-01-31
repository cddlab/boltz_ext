from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from scipy import optimize


def length(v: np.ndarray, eps: float = 1e-6) -> float:
    """Calculate the length of a vector."""
    return math.sqrt(max(eps, v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def unit_vec(v: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """Calculate the unit vector."""
    vl = length(v, eps=eps)
    return v / vl, vl


def calc_chiral_vol(crds: np.ndarray, iatm: int, aj: int) -> float:
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
    chiral: int
    w: float = 0.1
    slack: float = 0.05
    fmax: float = -100.0

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

    def print(self, crds: np.ndarray) -> None:
        """Print the chiral data."""
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        a2 = crds[self.aid2]
        a3 = crds[self.aid3]
        v1 = a1 - a0
        v2 = a2 - a0
        v3 = a3 - a0
        vol = np.dot(v1, np.cross(v2, v3))
        print(  # noqa: T201
            f"{self.aid0}-{self.aid1}-{self.aid2}-{self.aid3}:"
            f" {vol=:.2f} {self.chiral=:.2f}"
        )

    def calc(self, crds: np.ndarray) -> float:
        """Calculate the chiral data."""
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        a2 = crds[self.aid2]
        a3 = crds[self.aid3]
        v1 = a1 - a0
        v2 = a2 - a0
        v3 = a3 - a0
        vol = np.dot(v1, np.cross(v2, v3))

        if self.chiral > 0:  # noqa: SIM108
            thr = self.chiral - self.slack
        else:
            thr = self.chiral + self.slack

        delta = vol - thr
        ene = delta * delta * self.w

        if self.chiral > 0:
            if delta < 0:
                return ene
            else:  # noqa: RET505
                return 0
        else:  # noqa: PLR5501
            if delta > 0:
                return ene
            else:  # noqa: RET505
                return 0

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> bool:
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        a2 = crds[self.aid2]
        a3 = crds[self.aid3]
        v1 = a1 - a0
        v2 = a2 - a0
        v3 = a3 - a0
        vol = np.dot(v1, np.cross(v2, v3))

        if self.chiral > 0:
            thr = self.chiral - 0.5
        else:
            thr = self.chiral + 0.5

        delta = vol - thr
        dE = 2.0 * delta * self.w
        # print(f"   {dE=}")
        eps = 1e-2
        if thr < 0:
            dE = max(0, dE)
            if dE < eps:
                return False
        else:
            dE = min(0, dE)
            if dE > -eps:
                return False

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
    def calc_chiral_vol(iatm, mol, conf):
        aj = []
        ajname = []
        atom = mol.GetAtomWithIdx(iatm)
        for b in atom.GetBonds():
            j = b.GetOtherAtom(atom).GetIdx()
            aj.append(j)
            ajname.append(mol.GetAtomWithIdx(j).GetProp("name"))
        chiral_vol = calc_chiral_vol(conf.GetPositions(), iatm, aj)
        print(f"{chiral_vol=:.2f}")

        atom_name = atom.GetProp("name")
        chiral_tag = atom.GetChiralTag()
        print(f"{iatm=} {atom_name=} {chiral_tag=} {aj} {ajname}")

        return chiral_vol, aj[0], aj[1], aj[2]


@dataclass
class BondData:
    aid0: int
    aid1: int
    r0: float
    slack: float = 0
    w: float = 0.05
    fmax: float = 100.0

    def setup(self, ind, aid):
        if aid == 0:
            self.aid0 = ind
        elif aid == 1:
            self.aid1 = ind
        else:
            msg = f"Invalid data {ind=} {aid=}"
            raise ValueError(msg)

    def calc(self, crds: Any) -> float:
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        v1 = a0 - a1
        n1l = length(v1)

        r2 = self.r0 + self.slack
        r1 = self.r0 - self.slack
        if n1l > r2:
            delta = n1l - r2
        elif n1l < r1:
            delta = n1l - r1
        else:
            return 0
        ene = self.w * delta * delta
        return ene

    def grad(self, crds, grad):
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        v1 = a0 - a1
        n1l = length(v1)

        r2 = self.r0 + self.slack
        r1 = self.r0 - self.slack
        if n1l > r2:
            # delta = n1l - r2
            delta = r2 / n1l
        elif n1l < r1:
            # delta = n1l - r1
            delta = r1 / n1l
        else:
            return

        con = 2.0 * self.w * (1.0 - delta)
        grad[self.aid0] += v1 * con
        grad[self.aid1] -= v1 * con


_angl_patt = Chem.MolFromSmarts("*~*~*")


def get_angle_idxs(mol, base_id=0):
    ids = mol.GetSubstructMatches(_angl_patt)
    return np.asarray(ids) + base_id


@dataclass
class AngleData:
    aid0: int
    aid1: int
    aid2: int
    th0: float
    slack: float = math.radians(5.0)
    w: float = 0.05
    # fmax: float = 100.0

    def setup(self, ind, aid):
        if aid == 0:
            self.aid0 = ind
        elif aid == 1:
            self.aid1 = ind
        elif aid == 2:
            self.aid2 = ind
        else:
            raise ValueError(f"Invalid data {ind=} {aid=}")

    @staticmethod
    def calc_angle(ai, aj, ak, conf):
        crds = conf.GetPositions()
        eps = 1e-6
        theta, _, _, _, _, _ = AngleData._calc_angle_impl(ai, aj, ak, crds, eps)
        return theta

    @staticmethod
    def _calc_angle_impl(ai, aj, ak, crds, eps=1e-6):
        ri = crds[ai]
        rj = crds[aj]
        rk = crds[ak]

        rij = ri - rj
        rkj = rk - rj

        # distances/norm
        eij, Rij = unit_vec(rij, eps)
        ekj, Rkj = unit_vec(rkj, eps)

        # angle
        costh = eij.dot(ekj)
        costh = min(1.0, max(-1.0, costh))
        theta = math.acos(costh)

        return theta, costh, eij, Rij, ekj, Rkj

    def calc(self, crds):
        eps = 1e-6
        theta, _, _, _, _, _ = self._calc_angle_impl(
            self.aid0, self.aid1, self.aid2, crds, eps
        )

        th2 = self.th0 + self.slack
        th1 = self.th0 - self.slack

        if theta > th2:
            delta = theta - th2
        elif theta < th1:
            delta = theta - th1
        else:
            return 0

        ene = self.w * delta * delta
        return ene

    def grad(self, crds, grad):
        eps = 1e-6
        theta, costh, eij, Rij, ekj, Rkj = self._calc_angle_impl(
            self.aid0, self.aid1, self.aid2, crds, eps
        )

        th2 = self.th0 + self.slack
        th1 = self.th0 - self.slack

        if theta > th2:
            delta = theta - th2
        elif theta < th1:
            delta = theta - th1
        else:
            return

        # calc gradient
        df = 2.0 * self.w * delta

        sinth = math.sqrt(max(0.0, 1.0 - costh * costh))
        Dij = df / (max(eps, sinth) * Rij)
        Dkj = df / (max(eps, sinth) * Rkj)

        vec_dij = Dij * (costh * eij - ekj)
        vec_dkj = Dkj * (costh * ekj - eij)

        grad[self.aid0] += vec_dij
        grad[self.aid1] -= vec_dij
        grad[self.aid2] += vec_dkj
        grad[self.aid1] -= vec_dkj


class Restraints:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def make_bond(self, ai, aj, atoms, conf):
        crds = conf.GetPositions()
        v = crds[aj] - crds[ai]
        d = np.linalg.norm(v)
        bnd = BondData(ai, aj, d)
        self.bond_data.append(bnd)

        self.register_site(atoms[ai], lambda x: bnd.setup(x, 0))
        self.register_site(atoms[aj], lambda x: bnd.setup(x, 1))

    def make_link_bond(self, ai1, atoms1, ai2, atoms2, ideal):
        bnd = BondData(ai1, ai2, ideal)
        self.bond_data.append(bnd)

        self.register_site(atoms1[ai1], lambda x: bnd.setup(x, 0))
        self.register_site(atoms2[ai2], lambda x: bnd.setup(x, 1))

    def make_angle(self, ai, aj, ak, mol, conf, atoms):
        th0 = AngleData.calc_angle(ai, aj, ak, conf)
        angl = AngleData(ai, aj, ak, th0)
        self.angle_data.append(angl)
        # self.register_site(atoms[ai], (angl, 0))
        # self.register_site(atoms[aj], (angl, 1))
        # self.register_site(atoms[ak], (angl, 2))
        self.register_site(atoms[ai], lambda x: angl.setup(x, 0))
        self.register_site(atoms[aj], lambda x: angl.setup(x, 1))
        self.register_site(atoms[ak], lambda x: angl.setup(x, 2))

    def make_angle_restraints(self, mol, conf, atoms):
        idxs = get_angle_idxs(mol)
        for idx in idxs:
            ai, aj, ak = idx
            self.make_angle(ai, aj, ak, mol, conf, atoms)

    def make_chiral(self, iatm, mol, conf, atoms, invert=False):
        chiral_vol, aj0, aj1, aj2 = ChiralData.calc_chiral_vol(iatm, mol, conf)
        if invert:
            chiral_vol = -chiral_vol

        ch = ChiralData(iatm, aj0, aj1, aj2, chiral_vol)
        self.chiral_data.append(ch)
        # self.register_site(atoms, iatm, (ch, 0))
        # self.register_site(atoms, aj0, (ch, 1))
        # self.register_site(atoms, aj1, (ch, 2))
        # self.register_site(atoms, aj2, (ch, 3))

        self.register_site(atoms[iatm], lambda x: ch.setup(x, 0))
        self.register_site(atoms[aj0], lambda x: ch.setup(x, 1))
        self.register_site(atoms[aj1], lambda x: ch.setup(x, 2))
        self.register_site(atoms[aj2], lambda x: ch.setup(x, 3))

    def __init__(self):
        self.chiral_data = []
        self.bond_data = []
        self.angle_data = []
        self.sites = []
        # self.method = "BFGS"
        self.method = "CG"
        # self.method = "L-BFGS-B"

    def register_site2(self, get_func, set_func, i, value):
        sid = get_func(i)
        if sid == 0:
            self.sites.append([value])
            new_sid = len(self.sites)
            # atoms[i].restraint = new_sid
            set_func(i, new_sid)
        else:
            self.sites[sid - 1].append(value)

    def register_site(self, atom, value):
        sid = atom.restraint
        if sid == 0:
            self.sites.append([value])
            new_sid = len(self.sites)
            atom.restraint = new_sid
        else:
            self.sites[sid - 1].append(value)

    def get_sites(self, index):
        if index == 0:
            return None
        return self.sites[index - 1]

    def setup_site(self, ind, sid):
        if sid == 0:
            return
        sites = self.get_sites(sid)
        # for tgt, aid in sites:
        for tgt in sites:
            tgt(ind)
            # tgt.setup(ind, aid)

    def minimize(self, crds_in: torch.Tensor) -> None:
        """Minimize the restraints."""
        if len(self.chiral_data) == 0:
            return
        print("=== minimization ===")  # noqa: T201
        # print(f"{crds_in.shape=}")
        crds = crds_in.detach().cpu().numpy()
        crds = crds.reshape(-1)
        opt = optimize.minimize(
            self.calc, crds, jac=self.grad, method=self.method
        )  # , tol=1e-4)
        print(f"{opt=}")

        device = crds_in.device
        crds = opt.x.reshape(-1, 3)
        crds_in[:, :] = torch.tensor(crds).to(device)

        ch_ene = 0.0
        for ch in self.chiral_data:
            ch.print(crds)
            ch_ene += ch.calc(crds)
        print(f"chiral E={ch_ene}")
        b_ene = 0.0
        for b in self.bond_data:
            b_ene += b.calc(crds)
        print(f"bond E={b_ene}")
        a_ene = 0.0
        for a in self.angle_data:
            a_ene += a.calc(crds)
        print(f"angle E={a_ene}")

    def calc(self, crds_in):
        ene = 0.0
        crds = crds_in.reshape(-1, 3)
        # print(f"calc: {crds.shape=}")
        for ch in self.chiral_data:
            ene += ch.calc(crds)
        for b in self.bond_data:
            ene += b.calc(crds)
        for a in self.angle_data:
            ene += a.calc(crds)
        # print(f"calc: {ene=}")
        return ene

    def grad(self, crds_in):
        crds = crds_in.reshape(-1, 3)
        grad = np.zeros_like(crds)
        # print(f"grad: {crds.shape=}")
        # print(f"grad: {grad.shape=}")
        for ch in self.chiral_data:
            ch.grad(crds, grad)
        for b in self.bond_data:
            b.grad(crds, grad)
        for a in self.angle_data:
            a.grad(crds, grad)
        grad = grad.reshape(-1)
        return grad
