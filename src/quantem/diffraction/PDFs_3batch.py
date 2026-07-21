import numpy as np
import time

from scipy.spatial import KDTree
from datetime import timedelta

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np


class PDF3B(object):

    def __init__(self, device="cpu", v=1):
        if isinstance(device, int):
            self._xp = cp
            cp.cuda.runtime.setDevice(device)
        elif device.lower() == "cpu":
            self._xp = np
        self.device = device
        self._v = v
        return

    def pdf3B(
        self,
        atoms,
        dr=0.1,
        dtheta=1,
        hist_r_max=20,
        skip=1,
        batch_size=1,
        v=None,
        pbcs=None,
        subpixel_weights=True,
    ):
        xp = self._xp
        stime = time.perf_counter()
        v = self._v if v is None else v
        vprint = print if v >= 1 else lambda *a, **k: None
        if skip == None:
            skip = 1
        if pbcs is None:
            pbcs = atoms.get_pbc()
        self.pbcs = pbcs

        bbox_np = np.array(atoms.cell[(0, 1, 2), (0, 1, 2)])
        bbox = xp.array(bbox_np)
        pbcs_np = np.copy(pbcs)
        pbcs = xp.array(pbcs)
        positions = xp.array(atoms.positions)
        positions_np = np.array(atoms.positions)  # for KD tree

        vprint(f"Cell size (A): {bbox_np}")
        # vprint(f"max dist possible between two points with full pbcs: {np.sqrt(3*(bbox_np.max()/2)**2):.2f} A")
        vprint(f"Using a max radius of {hist_r_max} A")

        ### get center positions that aren't within r_max of a boundary without pbcs
        goods = np.ones(len(positions_np)).astype("bool")
        for i, p in enumerate(pbcs_np):
            if p:
                continue
            goods *= (positions_np[:, i] > hist_r_max) & (
                positions_np[:, i] < (bbox_np[i] - hist_r_max)
            )
        center_inds = np.where(goods)[0]

        ### KD tree for getting neighbors, cpu cuz faster
        start_KD = time.perf_counter()
        bbox_np_pbcs = bbox_np * pbcs_np
        if np.any(
            np.round(positions_np.max(axis=0), 9) >= np.round(bbox_np, 9)
        ):  # cant have atom right on cell edge
            for i in range(len(pbcs)):
                if bbox_np_pbcs[i] != 0:
                    bbox_np_pbcs += 1e-9
        tree_all = KDTree(positions_np, copy_data=True, boxsize=bbox_np_pbcs)
        center_inds_skip = center_inds[::skip]
        tree_centers = KDTree(
            positions_np[center_inds_skip], copy_data=True, boxsize=bbox_np_pbcs
        )
        inds_all = tree_centers.query_ball_tree(tree_all, hist_r_max)
        inds_all = [xp.array(i) for i in inds_all]
        end_KD = time.perf_counter()
        vprint(f"KDTree calc time: {end_KD - start_KD:.1e} s")

        ### Histogram coordinates
        hist_theta = xp.arange(0.0, 180 + dtheta, dtheta)
        hist_r = xp.arange(0.0, hist_r_max + dr, dr)
        num_bins_r = hist_r.shape[0]
        num_bins_theta = hist_theta.shape[0]
        hist_sig = xp.zeros((num_bins_theta, num_bins_r, num_bins_r))

        ### Density
        cell = xp.array(atoms.cell)
        volume = xp.abs(xp.sum(cell[:, 0] * xp.cross(cell[:, 1], cell[:, 2])))
        dens = positions.shape[0] / volume

        ### making big arrays that will be filled
        natoms_dens = (4 / 3 * xp.pi * hist_r_max**3) * dens
        maxval_n = int(1.25 * natoms_dens**2)
        vals = xp.zeros((3, maxval_n * batch_size)).astype("int")
        maxinds = max([len(i) for i in inds_all])
        inds2 = xp.zeros((batch_size, maxinds)).astype("int")
        if subpixel_weights:
            weights_full = xp.empty(maxval_n * batch_size)
            weights = weights_full
        else:
            weights_full = None
            weights = None

        if hist_r_max > 15:
            split_bincount = True
            off_inds = xp.triu_indices(maxinds, k=1)
        else:
            split_bincount = False
            off_inds = xp.where(~xp.eye(maxinds, dtype=bool))

        vprint(f"Num total atoms in sim = {len(positions)}")
        if not np.all(pbcs):
            vprint(f"Num non-edge atoms = {len(center_inds)}")
        if skip != 1:
            vprint(
                f"Skip = {skip}, so calculating using {len(center_inds_skip)} atoms as centers"
            )
        # vprint(f'atomic density = {dens:.3} atoms / A^3')
        vprint("Final shape will be: ", hist_sig.shape)

        for a0 in range(0, len(center_inds_skip), batch_size):
            b0 = min(a0 + batch_size, len(center_inds_skip))
            batch_center_inds = center_inds_skip[a0:b0]
            bsize = len(batch_center_inds)
            bnum_inds = [len(inds) for inds in inds_all[a0:b0]]
            for i in range(bsize):
                inds2[i, : bnum_inds[i]] = inds_all[a0 + i]

            vecs = self._get_vecs_pbcs(
                positions[batch_center_inds],
                positions[inds2[:bsize]],
                bbox=bbox,
                pbcs=pbcs,
            )
            for i in range(bsize):
                vecs[i, bnum_inds[i] :] = 0

            drs = xp.sqrt(xp.sum(vecs**2, axis=-1))

            thetas_full = self._get_thetas_all(vecs, vecs, deg=True)
            thetas = thetas_full[:, off_inds[0], off_inds[1]]
            thetas = thetas[:bsize].ravel()

            dr1s = drs[:, off_inds[0]]
            dr2s = drs[:, off_inds[1]]
            dr1s = dr1s[:bsize].ravel()
            dr2s = dr2s[:bsize].ravel()

            r_ind1 = dr1s / dr
            r_ind2 = dr2s / dr
            theta_ind = thetas / dtheta

            if subpixel_weights:
                ### this isn't perfect, doesn't do corners
                r_floor1 = xp.floor(r_ind1).astype("int")
                dr_ind1 = r_ind1 - r_floor1

                r_floor2 = xp.floor(r_ind2).astype("int")
                dr_ind2 = r_ind2 - r_floor2

                theta_floor = xp.floor(theta_ind).astype("int")
                dtheta_ind = theta_ind - theta_floor

            else:
                r_floor1 = xp.round(r_ind1).astype("int")
                r_floor2 = xp.round(r_ind2).astype("int")
                theta_floor = xp.round(theta_ind).astype("int")

            good_vals = (r_floor1 > 1e-9) & (r_floor2 > 1e-9) & (theta_floor > 0) & (theta_floor < 180)
            nvals = good_vals.sum()
            vals[0, :nvals] = theta_floor[good_vals]
            vals[1, :nvals] = r_floor2[good_vals]
            vals[2, :nvals] = r_floor1[good_vals]
            if subpixel_weights:
                weights_full[:nvals] = (
                    dtheta_ind[good_vals] + dr_ind2[good_vals] + dr_ind1[good_vals]
                ) / 3
                weights = weights_full[:nvals]

            ### Because taking triu, have to do this twice
            ### and doing bincount twice with half indices seems to be ~10% faster than
            ### once with twice as many (for a 20A cutoff), but slower if < 15 A
            hist_sig += self._bincountdd(
                vals[:, :nvals],
                (num_bins_theta, num_bins_r, num_bins_r),
                weights=weights,
            )
            if split_bincount:
                vals[[1,2]] = vals[[2,1]]
                hist_sig += self._bincountdd(
                    vals[:, :nvals],
                    (num_bins_theta, num_bins_r, num_bins_r),
                    weights=weights,
                )


        hist_sig /= len(center_inds_skip)

        gr = hist_sig.copy()
        r1 = hist_r[1:][None, None, ...]
        r2 = hist_r[1:][None, ..., None]
        th = hist_theta[1:-1, None, None]
        gr[1:-1, 1:, 1:] /= (
            dens**2
            * dr**2
            * 8 # if all off diag indices
            # * 4 # if triu
            * xp.pi**2
            * r1**2
            * r2**2
            * xp.sin(xp.deg2rad(th))
            * np.deg2rad(dtheta)
        )

        ### required because dividing by sin(theta)
        gr[-1] = 0
        gr[0] = 0
        gr[:, 0] = 0
        gr[:, :, 0] = 0

        vprint("-- done --")
        ttime = time.perf_counter() - stime
        vprint(
            f"Total time (h:m:s) {str(timedelta(seconds=round(ttime,3))).rstrip('0')}"
        )
        vprint(f"Center atoms per sec: {len(center_inds_skip) / ttime:.1f}\n")
        return gr

    def _get_vecs_pbcs(self, points, pointslists, bbox, pbcs=[1, 1, 1]):
        xp = self._xp
        if np.any(pbcs):
            assert xp.all(xp.min(pointslists, axis=1) >= 0)
            assert xp.all(xp.max(pointslists, axis=1) <= bbox)
        # assert xp.issubdtype(
        #     pointslists, float
        # ), f"pointslists type: {pointslists.dtype}"
        dif = pointslists - points[:, None]
        for i, ind in enumerate(pbcs):
            if ind:
                dif[:, :, i] = (
                    xp.mod(dif[:, :, i] + bbox[i] * 0.5, bbox[i]) - bbox[i] * 0.5
                )
        return dif

    def _bincountdd(self, vals, nbins, weights=None):
        # based on xp.histogramdd
        """
        vals: array shape (N, D)
        weights: array shape N
        nbins: (D,) tuple (nbinsz, nbinsy, nbinsx)
        based on xp.histogramdd
        """
        xp = self._xp
        vals = xp.floor(vals).astype("int")
        xy = xp.ravel_multi_index(vals, nbins, mode="wrap")
        minlength = int(np.prod(nbins))
        if weights is not None:
            hist = xp.bincount(xy, weights=1 - weights, minlength=minlength).reshape(
                nbins
            )
            xy = xp.ravel_multi_index(vals + 1, nbins, mode="wrap")
            hist += xp.bincount(xy, weights=weights, minlength=minlength).reshape(nbins)
        else:
            hist = xp.bincount(xy, minlength=minlength).reshape(nbins)
        return hist

    def _get_thetas_all(self, v1, v2, deg=True):
        """
        get angles between all vectors, outer product style
        """
        xp = self._xp
        v11 = self._norm_vecs(v1)
        v21 = self._norm_vecs(v2)
        mmul = xp.transpose(xp.matmul(v11, xp.transpose(v21, (0, 2, 1))), (0, 2, 1))
        theta = xp.arccos(xp.clip(mmul, -1.0, 1.0))
        if deg:
            theta = xp.rad2deg(theta)
        return theta

    def _norm_vecs(self, v):
        xp = self._xp
        if xp.ndim(v) == 1:
            return v / xp.sqrt(xp.sum(v**2, axis=-1))[..., None]
        mags = xp.sqrt(xp.sum(v**2, axis=-1))
        bads = xp.where(mags == 0)
        mags[bads] = 1
        normed = v / mags[..., None]
        normed[bads] = 0
        return normed
