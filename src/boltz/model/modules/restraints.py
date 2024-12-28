from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import optimize


def unit_vec(v):
    vl = np.linalg.norm(v)
    return v / vl, vl


def calc_chiral_vol(crds, iatm, aj):
    # crds = conf.GetPositions()
    vc = crds[iatm]
    v1 = crds[aj[0]] - vc
    v2 = crds[aj[1]] - vc
    v3 = crds[aj[2]] - vc
    # print(v1, v2, v3)
    vol = np.dot(v1, np.cross(v2, v3))
    return vol


@dataclass
class ChiralData:
    aid0: int
    aid1: int
    aid2: int
    aid3: int
    chiral: int
    w: float = 1.0
    fmax: float = 100.0

    # set0: bool = False
    # set1: bool = False
    # set2: bool = False
    # set3: bool = False

    def setup(self, ind, aid):
        if aid == 0:
            # assert not self.set0
            self.aid0 = ind
            # self.set0 = True
        elif aid == 1:
            # assert not self.set1
            self.aid1 = ind
            # self.set1 = True
        elif aid == 2:
            # assert not self.set2
            self.aid2 = ind
            # self.set2 = True
        elif aid == 3:
            # assert not self.set3
            self.aid3 = ind
            # self.set3 = True
        else:
            raise ValueError(f"Invalid data {ind=} {aid=}")

    def print(self, crds):
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        a2 = crds[self.aid2]
        a3 = crds[self.aid3]
        v1 = a1 - a0
        v2 = a2 - a0
        v3 = a3 - a0
        vol = np.dot(v1, np.cross(v2, v3))
        print(
            f"{self.aid0}-{self.aid1}-{self.aid2}-{self.aid3}: {vol=:.2f} {self.chiral=:.2f}"
        )

    def calc(self, crds):
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
        ene = delta * delta * self.w

        if self.chiral > 0:
            if delta < 0:
                return ene
            else:
                return 0
        else:
            if delta > 0:
                return ene
            else:
                return 0

    def grad(self, crds, grad):
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

        print(f"   force aver: {(n1l+n2l+n3l+ncl)/4}")

        if self.fmax > 0:
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

    def restr(self, crds_in):
        device = crds_in.device
        crds = crds_in.detach().cpu().numpy()
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

        # print(f"XXX:{self.aid0} {vol=:.2f} {self.chiral=:.2f} {delta=:.2f}")

        # dE = -sign * 2.0 * delta * self.w
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

        print(f"   force aver: {(n1l+n2l+n3l+ncl)/4}")

        n1l = min(n1l, self.fmax)
        n2l = min(n2l, self.fmax)
        n3l = min(n3l, self.fmax)
        ncl = min(ncl, self.fmax)

        f1 = -n1 * n1l
        f2 = -n2 * n2l
        f3 = -n3 * n3l
        fc = -nc * ncl

        # crds[self.aid0] += fc
        # crds[self.aid1] += f1
        # crds[self.aid2] += f2
        # crds[self.aid3] += f3
        # new_vol = calc_chiral_vol(crds, self.aid0, [self.aid1, self.aid2, self.aid3])
        # print(f"  {new_vol=:.2f} {self.chiral-new_vol=:.2f}")

        crds_in[self.aid0] += torch.tensor(fc).to(device)
        crds_in[self.aid1] += torch.tensor(f1).to(device)
        crds_in[self.aid2] += torch.tensor(f2).to(device)
        crds_in[self.aid3] += torch.tensor(f3).to(device)
        return True


# def decode_chiral_vol(vol):
#     if vol & 0x80:
#         vol = vol - 0x100
#     return vol / 40


# def decode_chiral(val: int, aid: int):
#     a0 = val & 0xFF
#     if a0 & 0x80:
#         a0 = a0 - 0x100
#     a1 = (val >> 8) & 0xFF
#     if a1 & 0x80:
#         a1 = a1 - 0x100
#     a2 = (val >> 16) & 0xFF
#     if a2 & 0x80:
#         a2 = a2 - 0x100
#     chiral = (val >> 24) & 0xFF
#     chvol = decode_chiral_vol(chiral)
#     print(f"{a0=}, {a1=}, {a2=}, {chvol=}")
#     return ChiralData(aid, aid + a0, aid + a1, aid + a2, chvol)


# def encode_aid(aid: int) -> int:
#     if aid < 0:
#         return (aid + 0x100) & 0xFF
#     else:
#         return aid & 0xFF


# def encode_chiral_vol(vol):
#     v = int(vol * 40)
#     v = max(-128, min(127, v))
#     chiral_vol = encode_aid()
#     return chiral_vol


# def encode_chiral(iatm, mol, conf):
#     # aj, chirality_type
#     aj = []
#     ajname = []
#     atom = mol.GetAtomWithIdx(iatm)
#     for b in atom.GetBonds():
#         j = b.GetOtherAtom(atom).GetIdx()
#         aj.append(j)
#         ajname.append(mol.GetAtomWithIdx(j).GetProp("name"))
#     # chirality_type = const.chirality_type_ids.get(
#     #     atom.GetChiralTag(), unk_chirality
#     # )
#     # chirality_type = encode_chiral(i, aj, chirality_type)
#     res = encode_aid(aj[0] - iatm)
#     res += encode_aid(aj[1] - iatm) * 0x100
#     res += encode_aid(aj[2] - iatm) * 0x10000
#     chiral_vol = calc_chiral_vol(conf.GetPositions(), iatm, aj)
#     print(f"{chiral_vol=:.2f}")
#     chiral_vol = encode_aid(int(chiral_vol * 40))
#     print(f"{decode_chiral_vol(chiral_vol)=}")
#     res += chiral_vol * 0x1000000

#     atom_name = atom.GetProp("name")
#     chiral_tag = atom.GetChiralTag()
#     print(f"{iatm=} {atom_name=} {chiral_tag=} {aj} {ajname}--> {res:X}")

#     return res


@dataclass
class BondData:
    aid0: int
    aid1: int
    r0: float
    slack: float = 0.5
    w: float = 0.1
    fmax: float = 100.0

    def setup(self, ind, aid):
        if aid == 0:
            self.aid0 = ind
        elif aid == 1:
            self.aid1 = ind
        else:
            raise ValueError(f"Invalid data {ind=} {aid=}")

    def calc(self, crds):
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        v1 = a0 - a1
        _, n1l = unit_vec(v1)

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
        _, n1l = unit_vec(v1)

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


class Restraints:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def make_chiral(iatm, mol, conf):
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

        ch = ChiralData(iatm, aj[0], aj[1], aj[2], chiral_vol)
        return ch

    def make_bond(self, ai, aj, mol, conf, atoms):
        crds = conf.GetPositions()
        v = crds[aj] - crds[ai]
        d = np.linalg.norm(v)
        bnd = BondData(ai, aj, d)
        self.bond_data.append(bnd)
        self.register_site(atoms, ai, (bnd, 0))
        self.register_site(atoms, aj, (bnd, 1))

        # return (bnd, 0), (bnd, 1)

    def __init__(self):
        self.chiral_data = []
        self.bond_data = []
        self.sites = []
        # self.method = "BFGS"
        # self.method = "CG"
        self.method = "L-BFGS-B"

    def add_chiral_data(self, ch):
        self.chiral_data.append(ch)
        # ind = len(self.chiral_data) - 1
        return (ch, 0), (ch, 1), (ch, 2), (ch, 3)
        # return (ind, 0), (ind, 1), (ind, 2), (ind, 3)
        # assert ind < (1 << 16)
        # return ind, ind + 1 * 0x10000, ind + 2 * 0x10000, ind + 3 * 0x10000

    def register_site(self, atoms, i, value):
        sid = atoms[i].chirality
        if sid == 0:
            self.sites.append([value])
            new_sid = len(self.sites)
            atoms[i].chirality = new_sid
        else:
            self.sites[sid - 1].append(value)

    # def register_site(self, index, value):
    #     if index == 0:
    #         self.sites.append([value])
    #         new_ind = len(self.sites)
    #         return new_ind
    #     else:
    #         self.sites[index - 1].append(value)
    #         return index

    def get_sites(self, index):
        if index == 0:
            return None
        return self.sites[index - 1]

    def setup_chiral_data(self, ind, val):
        if val == 0:
            return
        sites = self.get_sites(val)
        # for val0, aid in sites:
        #     print(f"{ind} {val=:X}: {val0=} {aid=}")
        #     ch = self.chiral_data[val0]
        for ch, aid in sites:
            ch.setup(ind, aid)

    def minimize(self, crds_in):
        # print(f"{crds_in.shape=}")
        crds = crds_in.detach().cpu().numpy()
        crds = crds.reshape(-1)
        opt = optimize.minimize(self.calc, crds, jac=self.grad, method=self.method)
        print(f"{opt=}")

        device = crds_in.device
        crds = opt.x.reshape(-1, 3)
        for ch in self.chiral_data:
            ch.print(crds)
        crds_in[:, :] = torch.tensor(crds).to(device)

    def calc(self, crds_in):
        ene = 0.0
        crds = crds_in.reshape(-1, 3)
        # print(f"calc: {crds.shape=}")
        for ch in self.chiral_data:
            ene += ch.calc(crds)
        for b in self.bond_data:
            ene += b.calc(crds)
        print(f"calc: {ene=}")
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
        grad = grad.reshape(-1)
        return grad
