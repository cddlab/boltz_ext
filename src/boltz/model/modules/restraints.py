from __future__ import annotations

import itertools

import numpy as np
import torch
from rdkit import Chem
from scipy import optimize
from .chiral_data import ChiralData, calc_chiral_vol
from .angle_restr_data import AngleData, get_angle_idxs
from .bond_restr_data import BondData


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
        # self.start_step = config.get("start_step", 50)
        # self.end_step = config.get("end_step", 999)
        self.start_sigma = config.get("start_sigma", 1.0)

        self.chiral_config = config.get("chiral", {})
        self.bond_config = config.get("bond", {})
        self.angle_config = config.get("angle", {})

        self.method = self.config.get("method", "CG")
        self.max_iter = int(self.config.get("max_iter", "100"))

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

    def make_angle_restraints(self, mol, conf, atoms, atom_names=None) -> None:
        idxs = get_angle_idxs(mol)
        if self.verbose:
            print(f"{idxs=}")
        for idx in idxs:
            ai, aj, ak = idx
            if atom_names is not None:
                an1 = mol.GetAtomWithIdx(int(ai)).GetProp("name")
                an2 = mol.GetAtomWithIdx(int(aj)).GetProp("name")
                an3 = mol.GetAtomWithIdx(int(ak)).GetProp("name")
                if (
                    an1 not in atom_names
                    or an2 not in atom_names
                    or an3 not in atom_names
                ):
                    print(f"skip {an1=} {an2=} {an3=}")
                    continue

            self.make_angle(ai, aj, ak, mol, conf, atoms)

    def make_chiral_impl(
        self, ai: int, aj: list[int], mol, conf, atoms, invert: bool = False
    ) -> None:
        chiral_vol = calc_chiral_vol(conf.GetPositions(), ai, aj)
        if invert:
            chiral_vol = -chiral_vol
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
            self.make_chiral_impl(iatm, cand, mol, conf, atoms, invert=invert)

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

        self.show_start()

    def show_start(self) -> None:
        """Show the start."""
        print("=== start restr ===")
        print(f"{self.method=} {self.max_iter=}")

    def print_stat_tensor(self, crds_in) -> None:
        crds = crds_in.detach().cpu().numpy()
        crds = crds[:, self.active_sites, :]
        self.print_stat(crds)

    def print_stat(self, crds) -> None:
        """Print the statistics."""
        nbatch = crds.shape[0]
        for i in range(nbatch):
            ch_ene = 0.0
            ch_sd = 0.0
            for ch in self.chiral_data:
                if self.verbose:
                    ch.print(crds[i])
                ch_sd += ch.calc_sd(crds[i])
                ch_ene += ch.calc(crds[i])
            print(f"chiral E={ch_ene:.5f}")
            ch_rmsd = np.sqrt(ch_sd/len(self.chiral_data))
            print(f"chiral rmsd={ch_rmsd:.5f}")

            b_ene = 0.0
            b_sd = 0.0
            for b in self.bond_data:
                if self.verbose:
                    b.print(crds[i])
                b_ene += b.calc(crds[i])
                b_sd += b.calc_sd(crds[i])
            print(f"bond E={b_ene:.5f}")
            b_rmsd = np.sqrt(b_sd/len(self.bond_data))
            print(f"bond rmsd={b_rmsd:.5f}")

            a_ene = 0.0
            a_sd = 0.0
            for a in self.angle_data:
                if self.verbose:
                    a.print(crds[i])
                a_ene += a.calc(crds[i])
                a_sd += a.calc_sd(crds[i])
            print(f"angle E={a_ene:.5f}")
            a_rmsd = np.sqrt(a_sd/len(self.angle_data))
            print(f"angle rmsd={a_rmsd:.5f}")


    def minimize(self, batch_crds_in: torch.Tensor, istep: int, sigma_t: float) -> None:
        """Minimize the restraints."""
        if sigma_t > self.start_sigma:
            return
        # if not (self.start_step <= istep < self.end_step):
        #     return

        if len(self.chiral_data) == 0:
            return

        device = batch_crds_in.device
        crds_in = batch_crds_in

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

        options = {"maxiter": self.max_iter}
        opt = optimize.minimize(
            self.calc, crds, jac=self.grad, method=self.method, options=options
        )
        # print(f"{opt=}")

        crds = opt.x.reshape(self.nbatch, self.natoms, 3)
        crds_in[:, self.active_sites, :] = torch.tensor(crds).to(device)

        if self.verbose:
            self.print_stat(crds)

    def finalize(self, batch_crds_in: torch.Tensor, istep: int) -> None:
        """Finalize the restraints."""
        if len(self.chiral_data) == 0:
            return
        print(f"=== final stats {istep} ===")
        self.print_stat_tensor(batch_crds_in)
            

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
