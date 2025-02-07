from __future__ import annotations

import math
import itertools
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
            f" {vol=:.2f} {self.chiral_vol=:.2f}"
        )

    def is_valid(self) -> bool:
        """Check if the chiral data is valid."""
        return self.aid0 >= 0 and self.aid1 >= 0 and self.aid2 >= 0 and self.aid3 >= 0

    def reset_indices(self) -> None:
        """Reset the indices."""
        self.aid0 = -1
        self.aid1 = -1
        self.aid2 = -1
        self.aid3 = -1

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
        if self.chiral_vol > 0:
            thr = self.chiral_vol - self.slack
        else:
            thr = self.chiral_vol + self.slack

        delta = vol - thr

        ene = delta * delta * self.w

        return ene
        # if self.chiral_vol > 0:
        #     if delta < 0:
        #         return ene
        #     else:  # noqa: RET505
        #         return 0
        # else:  # noqa: PLR5501
        #     if delta > 0:
        #         return ene
        #     else:  # noqa: RET505
        #         return 0

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> bool:
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


@dataclass
class BondData:
    """Class for bond data."""

    aid0: int
    aid1: int
    r0: float
    slack: float = 0
    w: float = 0.05
    # fmax: float = 100.0

    def is_valid(self) -> bool:
        """Check if the bond data is valid."""
        return self.aid0 >= 0 and self.aid1 >= 0

    def reset_indices(self) -> None:
        """Reset the indices."""
        self.aid0 = -1
        self.aid1 = -1

    def setup(self, ind: int, aid: int) -> None:
        """Set up bond data."""
        if aid == 0:
            self.aid0 = ind
        elif aid == 1:
            self.aid1 = ind
        else:
            msg = f"Invalid data {ind=} {aid=}"
            raise ValueError(msg)

    def calc(self, crds: np.ndarray) -> float:
        """Calculate the bond data."""
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

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> None:
        """Calculate the gradient."""
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


def get_angle_idxs(mol: Chem.Mol, base_id: int = 0) -> np.ndarray:
    """Get the angle indexes."""
    ids = mol.GetSubstructMatches(_angl_patt)
    return np.asarray(ids) + base_id


@dataclass
class AngleData:
    """Class for angle data."""

    aid0: int
    aid1: int
    aid2: int
    th0: float
    slack: float = math.radians(5.0)
    w: float = 0.05
    # fmax: float = 100.0

    def is_valid(self) -> bool:
        """Check if the angle data is valid."""
        return self.aid0 >= 0 and self.aid1 >= 0 and self.aid2 >= 0

    def reset_indices(self) -> None:
        """Reset the indices."""
        self.aid0 = -1
        self.aid1 = -1
        self.aid2 = -1

    def setup(self, ind: int, aid: int) -> None:
        """Set up the angle data."""
        if aid == 0:
            self.aid0 = ind
        elif aid == 1:
            self.aid1 = ind
        elif aid == 2:
            self.aid2 = ind
        else:
            raise ValueError(f"Invalid data {ind=} {aid=}")

    @staticmethod
    def calc_angle(ai: int, aj: int, ak: int, conf) -> float:
        """Calculate the angle."""
        crds = conf.GetPositions()
        eps = 1e-6
        theta, _, _, _, _, _ = AngleData._calc_angle_impl(ai, aj, ak, crds, eps)
        return theta

    @staticmethod
    def _calc_angle_impl(
        ai: int, aj: int, ak: int, crds: np.ndarray, eps: float = 1e-6
    ) -> tuple[float, float, np.ndarray, float, np.ndarray, float]:
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

    def calc(self, crds: np.ndarray) -> float:
        """Calculate the angle energy."""
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

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> None:
        """Calculate the gradient."""
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
    """Class for restraints."""

    _instance = None

    @classmethod
    def get_instance(cls) -> Restraints:
        """Get the instance of the restraints."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.chiral_data = []
        self.bond_data = []
        self.angle_data = []
        self.sites = []

    def set_config(self, config: dict) -> None:
        """Set the configuration."""
        self.config = config

        self.verbose = config.get("verbose", False)
        self.start_step = config.get("start_step", 50)
        self.end_step = config.get("end_step", 999)

        self.chiral_config = config.get("chiral", {})
        self.bond_config = config.get("bond", {})
        self.angle_config = config.get("angle", {})

    def _create_bond_data(self, d: float) -> BondData:
        return BondData(
            -1,
            -1,
            d,
            w=self.bond_config.get("weight", 0.05),
            slack=self.bond_config.get("slack", 0),
        )

    def _create_angle_data(self, th0: float) -> AngleData:
        return AngleData(
            -1,
            -1,
            -1,
            th0,
            w=self.angle_config.get("weight", 0.05),
            slack=self.angle_config.get("slack", 0),
        )

    def _create_chiral_data(self, chiral_vol: float) -> ChiralData:
        return ChiralData(
            -1,
            -1,
            -1,
            -1,
            chiral_vol,
            w=self.chiral_config.get("weight", 0.05),
            slack=self.chiral_config.get("slack", 0),
            fmax=self.chiral_config.get("f_max", 0),
        )

    def make_bond(self, ai: int, aj: int, atoms, conf) -> None:
        """Make bond data."""
        crds = conf.GetPositions()
        v = crds[aj] - crds[ai]
        d = np.linalg.norm(v)
        bnd = self._create_bond_data(d)
        self.bond_data.append(bnd)

        self.register_site(atoms[ai], lambda x: bnd.setup(x, 0))
        self.register_site(atoms[aj], lambda x: bnd.setup(x, 1))

    def make_link_bond(self, ai1: int, atoms1, ai2: int, atoms2, ideal: float) -> None:
        """Make link bond."""
        bnd = self._create_bond_data(ideal)
        self.bond_data.append(bnd)

        self.register_site(atoms1[ai1], lambda x: bnd.setup(x, 0))
        self.register_site(atoms2[ai2], lambda x: bnd.setup(x, 1))

    def _get_parsed_atom(self, chains, keys):
        cnam, ires, anam = keys
        if cnam not in chains:
            print(f"{cnam=} not found in chains")
            return None, None
        res = None
        for r in chains[cnam].residues:
            if r.idx == ires - 1:
                res = r
                break
        if res is None:
            print(f"{ires=} not found in {cnam=}")
            return None, None

        for i, a in enumerate(res.atoms):
            if a.name == anam:
                return i, res.atoms

        print(f"{anam=} not found in {cnam=}, {ires=}")
        return None, None

    def link_bonds_by_conf(self, chains, config) -> None:
        """Make link bonds by config."""
        # print(f"{chains=}")
        # print(f"{config=}")
        for entry in config:
            if "bond" not in entry:
                continue
            bond_cfg = entry["bond"]
            atom1 = bond_cfg["atom1"]
            atom2 = bond_cfg["atom2"]
            r0 = bond_cfg["r0"]

            ai1, atoms1 = self._get_parsed_atom(chains, atom1)
            if ai1 is None:
                print(f"{atom1=} not found")
                continue
            ai2, atoms2 = self._get_parsed_atom(chains, atom2)
            if ai2 is None:
                print(f"{atom2=} not found")
                continue
            self.make_link_bond(ai1, atoms1, ai2, atoms2, r0)

    def make_angle(self, ai, aj, ak, mol, conf, atoms) -> None:
        """Make angle data."""
        th0 = AngleData.calc_angle(ai, aj, ak, conf)
        angl = self._create_angle_data(th0)
        self.angle_data.append(angl)
        self.register_site(atoms[ai], lambda x: angl.setup(x, 0))
        self.register_site(atoms[aj], lambda x: angl.setup(x, 1))
        self.register_site(atoms[ak], lambda x: angl.setup(x, 2))

    def make_angle_restraints(self, mol, conf, atoms) -> None:
        idxs = get_angle_idxs(mol)
        for idx in idxs:
            ai, aj, ak = idx
            self.make_angle(ai, aj, ak, mol, conf, atoms)

    def make_chiral_impl(self, ai: int, aj: list[int], mol, conf, atoms):
        chiral_vol = calc_chiral_vol(conf.GetPositions(), ai, aj)
        ch = self._create_chiral_data(chiral_vol)
        self.chiral_data.append(ch)

        self.register_site(atoms[ai], lambda x: ch.setup(x, 0))
        self.register_site(atoms[aj[0]], lambda x: ch.setup(x, 1))
        self.register_site(atoms[aj[1]], lambda x: ch.setup(x, 2))
        self.register_site(atoms[aj[2]], lambda x: ch.setup(x, 3))
        print(f"chiral restr {ai} - {aj}: vol={chiral_vol:.2f}")

    def make_chiral(self, iatm: int, mol, conf, atoms, invert: bool = False) -> None:
        nei_ind = ChiralData.get_nei_atoms(iatm, mol)
        for cand in itertools.combinations(nei_ind, 3):
            self.make_chiral_impl(iatm, cand, mol, conf, atoms)

    def register_site(self, atom, value):
        sid = atom.restraint
        if sid == 0:
            self.sites.append([value])
            new_sid = len(self.sites)
            atom.restraint = new_sid
        else:
            self.sites[sid - 1].append(value)

    def get_sites(self, index: int):
        """Register the site."""
        if index == 0:
            return None
        return self.sites[index - 1]

    def setup_site(self, feat_restr_in: torch.Tensor) -> None:
        """Set up the restraintsites."""
        self.reset_indices()
        feat_restr = feat_restr_in[0].detach().cpu().numpy()

        self.active_sites = []
        for ind in range(len(feat_restr)):
            sid = int(feat_restr[ind])
            if sid == 0:
                continue
            self.active_sites.append(ind)
        print(f"{self.active_sites=}")

        for i, ind in enumerate(self.active_sites):
            sid = int(feat_restr[ind])
            if sid == 0:
                continue
            sites = self.get_sites(sid)
            for tgt in sites:
                tgt(i)
                # tgt(ind)

        for ch in self.chiral_data:
            if ch.is_valid():
                print(f"{ch.aid0}-{ch.aid1}-{ch.aid2}-{ch.aid3}")

    def minimize(self, batch_crds_in: torch.Tensor, istep: int) -> None:
        """Minimize the restraints."""
        if not (self.start_step <= istep < self.end_step):
            return

        if len(self.chiral_data) == 0:
            return

        device = batch_crds_in.device
        crds_in = batch_crds_in
        method = self.config.get("method", "CG")
        maxiter = int(self.config.get("maxiter", "100"))

        if self.verbose:
            print(f"=== minimization {istep} ===")  # noqa: T201
        crds = crds_in.detach().cpu().numpy()
        crds = crds[:, self.active_sites, :]
        self.nbatch = crds.shape[0]
        self.natoms = crds.shape[1]
        # print(f"{self.nbatch=}")
        # print(f"{self.natoms=}")
        # print(f"{crds.shape=}")
        crds = crds.reshape(-1)

        opt = optimize.minimize(
            self.calc, crds, jac=self.grad, method=method, options={"maxiter": maxiter}
        )
        # print(f"{opt=}")

        crds = opt.x.reshape(self.nbatch, self.natoms, 3)
        crds_in[:, self.active_sites, :] = torch.tensor(crds).to(device)

        if self.verbose:
            for i in range(self.nbatch):
                ch_ene = 0.0
                for ch in self.chiral_data:
                    ch.print(crds[i])
                    ch_ene += ch.calc(crds[i])
                print(f"chiral E={ch_ene}")
                b_ene = 0.0
                for b in self.bond_data:
                    b_ene += b.calc(crds[i])
                print(f"bond E={b_ene}")
                a_ene = 0.0
                for a in self.angle_data:
                    a_ene += a.calc(crds[i])
                print(f"angle E={a_ene}")

        # for ind in range(nbatch):
        #     crds_in = batch_crds_in[ind]
        #     if len(self.chiral_data) == 0:
        #         return
        #     if self.verbose:
        #         print(f"=== minimization {istep} ===")  # noqa: T201
        #     crds = crds_in.detach().cpu().numpy()
        #     crds = crds[self.active_sites, :]
        #     # print(f"{crds.shape=}")
        #     crds = crds.reshape(-1)
        #     method = self.config.get("method", "CG")

        #     opt = optimize.minimize(
        #         self.calc, crds, jac=self.grad, method=method
        #     )  # , tol=1e-4)
        #     # print(f"{opt=}")

        #     device = crds_in.device
        #     crds = opt.x.reshape(-1, 3)
        #     # crds_in[:, :] = torch.tensor(crds).to(device)
        #     crds_in[self.active_sites, :] = torch.tensor(crds).to(device)

    def calc(self, crds_in: np.ndarray) -> float:
        """Calculate energy."""
        ene = 0.0
        crds = crds_in.reshape(self.nbatch, self.natoms, 3)
        # print(f"calc: {crds.shape=}")
        for i in range(self.nbatch):
            for ch in self.chiral_data:
                if ch.is_valid():
                    ene += ch.calc(crds[i])
            for b in self.bond_data:
                if b.is_valid():
                    ene += b.calc(crds[i])
            for a in self.angle_data:
                if a.is_valid():
                    ene += a.calc(crds[i])
        # print(f"calc: {ene=}")
        return ene

    def grad(self, crds_in: np.ndarray) -> np.ndarray:
        """Calculate gradient."""
        crds = crds_in.reshape(self.nbatch, self.natoms, 3)
        grad = np.zeros_like(crds)
        # print(f"grad: {crds.shape=}")
        # print(f"grad: {grad.shape=}")
        for i in range(self.nbatch):
            for ch in self.chiral_data:
                if ch.is_valid():
                    ch.grad(crds[i], grad[i])
            for b in self.bond_data:
                if b.is_valid():
                    b.grad(crds[i], grad[i])
            for a in self.angle_data:
                if a.is_valid():
                    a.grad(crds[i], grad[i])
        grad = grad.reshape(-1)
        return grad

    def reset_indices(self) -> None:
        """Reset all restr indices."""
        for ch in self.chiral_data:
            ch.reset_indices()
        for b in self.bond_data:
            b.reset_indices()
        for a in self.angle_data:
            a.reset_indices()
