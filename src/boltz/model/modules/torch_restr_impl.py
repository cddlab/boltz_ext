from __future__ import annotations

import torch
from .bond_restr_data import BondData
from .angle_restr_data import AngleData
from .chiral_data import ChiralData
from torchmin.function import sf_value, ScalarFunction, de_value


def calculate_distances(atom_pos: torch.Tensor, atom_idx: torch.Tensor):
    # print(f"{atom_idx[:, 0].shape=}")
    # print(f"{atom_idx[:, 1].shape=}")

    # print(f"{atom_pos.shape=}")
    # print(f"{atom_pos[atom_idx[:, 0]].shape=}")
    # print(f"{atom_pos[atom_idx[:, 1]].shape=}")

    dir_vec = atom_pos[atom_idx[:, 0]] - atom_pos[atom_idx[:, 1]]
    dist = torch.norm(dir_vec, dim=1)
    unit_vec = dir_vec / dist.unsqueeze(1)
    return dist, unit_vec, dir_vec


class RestrTorchImpl:
    def __init__(
        self,
        bond_data: list[BondData],
        angle_data: list[AngleData],
        chiral_data: list[ChiralData],
        nbatch: int,
        natoms: int,
        device: torch.device | str,
    ):
        self.device = device
        self.nbatch = nbatch
        self.natoms = natoms

        self.setup_bonds(bond_data, nbatch, natoms)
        self.setup_angles(angle_data, nbatch, natoms)
        self.setup_chirals(chiral_data, nbatch, natoms)

    def setup_bonds(
        self,
        bond_data: list[BondData],
        nbatch: int,
        natoms: int,
    ) -> None:
        """Prepare bond indices and r0s for the given bond data."""
        data = []
        r0s = []
        for ib in range(nbatch):
            for bond in bond_data:
                if not bond.is_valid():
                    continue
                aid0 = bond.aid0 + ib * natoms
                aid1 = bond.aid1 + ib * natoms
                data.append([aid0, aid1])
                r0s.append(bond.r0)

        self.bond_idx = torch.tensor(data, dtype=torch.long, device=self.device)
        self.bond_r0s = torch.tensor(r0s, dtype=torch.float32, device=self.device)
        self.bond_k = bond_data[0].w

        # print(f"Bond indices: {atom_idx=}")
        # print(f"Bond r0s: {gpu_r0s=}")
        # return atom_idx, gpu_r0s

    def setup_angles(
        self,
        angle_data: list[AngleData],
        nbatch: int,
        natoms: int,
    ) -> None:
        """Prepare bond indices and r0s for the given bond data."""
        device = self.device
        data = []
        r0s = []
        for ib in range(nbatch):
            for angle in angle_data:
                if not angle.is_valid():
                    continue
                aid0 = angle.aid0 + ib * natoms
                aid1 = angle.aid1 + ib * natoms
                aid2 = angle.aid2 + ib * natoms
                data.append([aid0, aid1, aid2])
                r0s.append(angle.th0)

        self.angle_idx = torch.tensor(data, dtype=torch.long, device=device)
        self.angle_r0s = torch.tensor(r0s, dtype=torch.float32, device=device)
        self.angle_k = angle_data[0].w
        # print(f"Bond indices: {atom_idx=}")
        # print(f"Bond r0s: {gpu_r0s=}")
        # return atom_idx, gpu_r0s

    def setup_chirals(
        self,
        ch_data: list[ChiralData],
        nbatch: int,
        natoms: int,
    ) -> None:
        """Prepare bond indices and r0s for the given bond data."""
        device = self.device
        data = []
        r0s = []
        for ib in range(nbatch):
            for ch in ch_data:
                if not ch.is_valid():
                    continue
                aid0 = ch.aid0 + ib * natoms
                aid1 = ch.aid1 + ib * natoms
                aid2 = ch.aid2 + ib * natoms
                aid3 = ch.aid3 + ib * natoms
                data.append([aid0, aid1, aid2, aid3])
                r0s.append(ch.chiral_vol)

        self.chiral_idx = torch.tensor(data, dtype=torch.long, device=device)
        self.chiral_r0s = torch.tensor(r0s, dtype=torch.float32, device=device)
        self.chiral_k = ch_data[0].w
        # print(f"Chiral indices: {atom_idx=}")
        # print(f"Chiral r0s: {gpu_r0s=}")
        # return atom_idx, gpu_r0s

    def calc_bond_grad(
        self,
        atom_pos: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the bond force based on the positions and indices of atoms."""
        dist, unit_vec, _ = calculate_distances(atom_pos, self.bond_idx)

        x = dist - self.bond_r0s
        pot = self.bond_k * x * x
        force = 2.0 * self.bond_k * x

        forcevec = unit_vec * force[:, None]
        grad.index_add_(0, self.bond_idx[:, 0], forcevec)
        grad.index_add_(0, self.bond_idx[:, 1], -forcevec)

        return pot.sum()

    def calc_angle_grad(
        self,
        atom_pos: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """Calc angle grad."""
        _, _, r21 = calculate_distances(atom_pos, self.angle_idx[:, [0, 1]])
        _, _, r23 = calculate_distances(atom_pos, self.angle_idx[:, [2, 1]])

        dotprod = torch.sum(r23 * r21, dim=1)
        norm23inv = 1 / torch.norm(r23, dim=1)
        norm21inv = 1 / torch.norm(r21, dim=1)

        cos_theta = dotprod * norm21inv * norm23inv
        cos_theta = torch.clamp(cos_theta, -1, 1)
        theta = torch.acos(cos_theta)

        delta_theta = theta - self.angle_r0s
        pot = self.angle_k * delta_theta * delta_theta

        force0, force1, force2 = None, None, None

        sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
        coef = torch.zeros_like(sin_theta)
        nonzero = sin_theta != 0
        coef[nonzero] = 2.0 * self.angle_k * delta_theta[nonzero] / sin_theta[nonzero]
        force0 = (
            coef[:, None]
            * (cos_theta[:, None] * r21 * norm21inv[:, None] - r23 * norm23inv[:, None])
            * norm21inv[:, None]
        )
        force2 = (
            coef[:, None]
            * (cos_theta[:, None] * r23 * norm23inv[:, None] - r21 * norm21inv[:, None])
            * norm23inv[:, None]
        )
        force1 = -(force0 + force2)

        grad.index_add_(0, self.angle_idx[:, 0], force0)
        grad.index_add_(0, self.angle_idx[:, 1], force1)
        grad.index_add_(0, self.angle_idx[:, 2], force2)

        return pot.sum()

    def calc_chiral_grad(
        self,
        atom_pos: torch.Tensor,
        grad: torch.Tensor,
    ) -> None:
        """Calc chiral grad."""
        _, _, r21 = calculate_distances(atom_pos, self.chiral_idx[:, [0, 1]])
        a0 = atom_pos[self.chiral_idx[:, 0]]
        a1 = atom_pos[self.chiral_idx[:, 1]]
        a2 = atom_pos[self.chiral_idx[:, 2]]
        a3 = atom_pos[self.chiral_idx[:, 3]]
        v1 = a1 - a0
        v2 = a2 - a0
        v3 = a3 - a0

        # vol = torch.dot(v1, torch.cross(v2, v3, dim=1))
        vol = (v1 * torch.cross(v2, v3, dim=1)).sum(dim=1)

        delta = vol - self.chiral_r0s
        pot = self.chiral_k * delta * delta
        dE = 2.0 * self.chiral_k * delta
        dE = dE[:, None]

        f1 = torch.cross(v2, v3, dim=1) * dE
        f2 = torch.cross(v3, v1, dim=1) * dE
        f3 = torch.cross(v1, v2, dim=1) * dE
        fc = -f1 - f2 - f3

        grad.index_add_(0, self.chiral_idx[:, 0], fc)
        grad.index_add_(0, self.chiral_idx[:, 1], f1)
        grad.index_add_(0, self.chiral_idx[:, 2], f2)
        grad.index_add_(0, self.chiral_idx[:, 3], f3)

        return pot.sum()

    def grad(self, crds):
        device = self.device
        # gpu_crds = torch.tensor(crds).to(device).reshape(-1, 3)
        # print(f"{crds.shape=}")
        gpu_crds = crds.reshape(-1, 3)
        gpu_grad = torch.zeros_like(gpu_crds, device=device)

        f = self.calc_bond_grad(gpu_crds, gpu_grad)
        f += self.calc_angle_grad(gpu_crds, gpu_grad)
        f += self.calc_chiral_grad(gpu_crds, gpu_grad)

        # print(f"{f.shape=}")
        gpu_grad = gpu_grad.reshape(-1)
        # print(f"{gpu_grad.shape=}")
        return gpu_grad, f


class MyScalarFunc(ScalarFunction):
    def __init__(self, impl, x_shape):
        super().__init__(lambda x: x, x_shape)
        self.impl = impl

    def closure(self, x):
        # print(f"{x.shape=}")
        grad, f = self.impl.grad(x)
        return sf_value(f=f.detach(), grad=grad.detach(), hessp=None, hess=None)

    def dir_evaluate(self, x, t, d):
        x = x + d.mul(t)
        x = x.detach()
        grad, f = self.impl.grad(x)

        return de_value(f=float(f), grad=grad)
