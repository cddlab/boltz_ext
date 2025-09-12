from __future__ import annotations

import torch
from .bond_restr_data import BondData
from .angle_restr_data import AngleData
from .chiral_data import ChiralData
from torchmin.function import sf_value, ScalarFunction, de_value
import torch_cluster
import numpy as np
from rdkit import Chem


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

        self.vdw_dmax = 5.0

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

    def setup_vdw_radii(self, elems_oh):
        elems = torch.argmax(elems_oh[0], dim=-1).cpu().numpy()  # (nbatch, natoms,)
        # print(f"VdW elems: {elems.shape=}")
        # print(f"VdW elems: {elems=}")
        peri = Chem.rdchem.GetPeriodicTable()

        def elem2rvdw(x):
            try:
                if x < 1 or x > 118:
                    return 0.0
                return peri.GetRvdw(int(x))
            except Exception as e:
                print(f"elem2rvdw: {x}: {e}")
                return 0.0

        vdwr = np.vectorize(elem2rvdw)(elems)
        # print(f"{vdwr.shape=}")
        # print(f"{vdwr=}")
        return torch.tensor(vdwr, device=elems_oh.device, dtype=torch.float)

    def setup_vdw(
        self,
        nbatch: int,
        natoms: int,
        atom_mask,
        ligand_atoms: list[int],
        elems: torch.Tensor,
        config: dict,
    ) -> None:
        self.vdw_k = config.get("weight", None)
        if self.vdw_k is None or self.vdw_k <= 0:
            self.use_vdw = False
        else:
            self.use_vdw = True

        device = self.device
        atoms = np.arange(natoms)[atom_mask[0].bool().cpu().numpy()]

        self.ligand_idx = torch.tensor(ligand_atoms, device=device, dtype=torch.long)
        self.prot_idx = torch.tensor(
            [i for i in atoms if i not in ligand_atoms],
            device=device,
            dtype=torch.long,
        )
        self.lig_batch = torch.arange(nbatch, device=device).repeat_interleave(
            len(self.ligand_idx)
        )
        self.prot_batch = torch.arange(nbatch, device=device).repeat_interleave(
            len(self.prot_idx)
        )
        # print(f"{self.lig_batch=}")
        # print(f"{self.prot_batch=}")

        self.lind_flat = self.ligand_idx.repeat(nbatch) + self.lig_batch * natoms
        self.pind_flat = self.prot_idx.repeat(nbatch) + self.prot_batch * natoms

        if self.use_vdw:
            vdwr = self.setup_vdw_radii(elems)
            self.vdwr = vdwr.repeat(nbatch)
            # print(f"{self.vdwr.shape=}")

            self.vdw_scale = config.get("scale", 0.75)
            self.vdw_dmax = config.get("dmax", 5.0)
            self.vdw_lig_only = config.get("ligand_only", False)
            print(
                f"Use VdW restr scale={self.vdw_scale}, dmax={self.vdw_dmax},"
                f" ligand_only={self.vdw_lig_only}"
            )

    def update_vdw_idx(self, crds: torch.Tensor) -> None:
        """Update the vdw contact indices based on the current coordinates."""
        if not self.use_vdw:
            return
        vdw_thr = self.vdw_dmax
        # print(f"{crds.shape=} {crds.device=}")
        prot_crds = crds[:, self.prot_idx, :].reshape(-1, 3)
        lig_crds = crds[:, self.ligand_idx, :].reshape(-1, 3)

        idx_j, idx_i = torch_cluster.radius(
            x=prot_crds,
            y=lig_crds,
            batch_x=self.prot_batch,
            batch_y=self.lig_batch,
            r=vdw_thr,
        )
        if len(idx_i) > 0:
            # i: protein / j: ligans
            idx_i = self.pind_flat[idx_i]
            idx_j = self.lind_flat[idx_j]

            # print(f"{self.prot_idx=}")
            # print(f"{max(idx_i)=}")

            # print(f"{self.ligand_idx=}")
            # print(f"{max(idx_j)=}")

            vdw_idx = torch.stack([idx_i, idx_j], dim=1)
            self.vdw_idx = vdw_idx

            lig_vdwr_cur = self.vdwr[vdw_idx[:, 1]]
            # print(f"{vdw_idx[:, 1]=}")
            # print(f"{lig_vdwr_cur=}")
            prot_vdwr_cur = self.vdwr[vdw_idx[:, 0]]
            # print(f"{vdw_idx[:, 0]=}")
            # print(f"{prot_vdwr_cur=}")
            self.vdw_r0s = (prot_vdwr_cur + lig_vdwr_cur) * self.vdw_scale
            # print(f"{self.vdw_r0s=}")
        else:
            self.vdw_idx = None

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

    def calc_vdw_grad(
        self,
        atom_pos: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the vdw contact gradients."""
        dist, unit_vec, _ = calculate_distances(atom_pos, self.vdw_idx)

        # print(f"{dist.shape=}")
        # print(f"{self.vdw_r0s.shape=}")
        x = dist - self.vdw_r0s
        flag = x < 0
        pot = self.vdw_k * x * x * flag
        force = 2.0 * self.vdw_k * x * flag

        forcevec = unit_vec * force[:, None]
        # protein
        if not self.vdw_lig_only:
            grad.index_add_(0, self.vdw_idx[:, 0], forcevec)
        # ligand
        grad.index_add_(0, self.vdw_idx[:, 1], -forcevec)

        return pot.sum()

    def print_vdw_stat(
        self,
        atom_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Show the vdw contact status."""
        if self.vdw_idx is None:
            return
        atom_pos = atom_pos.reshape(-1, 3)

        idx = (self.vdw_idx // self.natoms)[:, 0]
        dist, unit_vec, _ = calculate_distances(atom_pos, self.vdw_idx)

        x = dist - self.vdw_r0s
        flag = x < 0

        for ib in range(self.nbatch):
            print(f"Protein-Ligand contacts: batch {ib}")
            d = dist[idx == ib]
            f = flag[idx == ib]
            print(f"  num of d < {self.vdw_scale}: {int(f.sum())}")
            print(f"  min dist: {float(d.min())}")

    def grad(self, crds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        # gpu_crds = torch.tensor(crds).to(device).reshape(-1, 3)
        # print(f"{crds.shape=}")
        gpu_crds = crds.reshape(-1, 3)
        gpu_grad = torch.zeros_like(gpu_crds, device=device)

        f = self.calc_bond_grad(gpu_crds, gpu_grad)
        f += self.calc_angle_grad(gpu_crds, gpu_grad)
        f += self.calc_chiral_grad(gpu_crds, gpu_grad)

        if self.use_vdw and self.vdw_idx is not None:
            f += self.calc_vdw_grad(gpu_crds, gpu_grad)

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
