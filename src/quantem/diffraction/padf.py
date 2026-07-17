import matplotlib.pyplot as plt
import torch
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import legval
from scipy.special import spherical_jn

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.polar_transform import (
    find_origin_angular_descent,
    find_origin_angular_grid,
    polar_transform,
)

from quantem.core.io.serialize import load
from pathlib import Path

class PairAngleDistributionFunction(AutoSerialize):
    """
    Compute pair-angle distribution function for a given 4D-STEM dataset.

    Run the following pipeline:
    pad = PairAngleDistributionFunction(ds)
    1. find_origin_angular_grid(ds)
    2. polar_transform(ds, origin_array=self.origin)
    3. rescale_intensity(self.polar, rho=1, fq=1)
    4. compute_avg_angular_correlation(self.polar_rescaled, dtype=torch.float64)
    5. extract_Bl_matrices(self.ang_corr)
    6. transform_to_real_space(self.Bl_mats[0], self.Bl_mats[1])
    7. reconstruct_PADF(self.real_Bl_mats, self.Bl_mats[1], self.ds.shape[0] * self.ds.shape[1])

    """
    def __init__(self,
                 ds: Dataset4dstem = None,
                 origin: NDArray | None = None,
                 polar_ds: Polar4dstem | None = None,
                 ang_corr: torch.Tensor | None = None,
                 Bl_mats = None
                 ):
        super().__init__()

        self.ds = ds
        self.origin = origin if origin is not None else find_origin_angular_grid(ds)
        self.polar = polar_ds if polar_ds is not None else polar_transform(ds, origin_array=self.origin)

        # TODO: we get rho and fq from sample identity, so either the user provides them, we find them in the dataset, or we look them up based on sample identity
        self.polar_rescaled = self.rescale_intensity(self.polar, rho=1, fq=1)

        self.ang_corr = ang_corr if ang_corr is not None else self.compute_avg_angular_correlation(self.polar_rescaled, dtype=torch.float64)
        self.Bl_mats = Bl_mats if Bl_mats is not None else self.extract_Bl_matrices(self.ang_corr)
        self.real_Bl_mats = self.transform_to_real_space(self.Bl_mats[0], self.Bl_mats[1])
        self.padf = self.reconstruct_PADF(self.real_Bl_mats, self.Bl_mats[1], 20) # TODO: determine atoms in beam
    
    def rescale_intensity(self,
                          data: Polar4dstem | Dataset4dstem | None = None,
                          rho: float | int | None = None,
                          fq: float | int | None = None):
        """
        Equation 7: Take the raw intensity and divide by the following factors
        - Mean number density (rho)
        - Mean atomic scattering factor squared (f(q)^2)
        - Number of atoms in the beam (N_a)
        - phi_0 (dependent on experimental parameters)

        Most likely we will need to fit phi_0 * N_a term via another function
        """
        # TODO: Look at the Martin code to see how he calculated atoms in beam and phi_0
        intensity = data.tensor
        rescaled_intensity = intensity / (rho * fq**2) # / still need to include atoms in beam and phi
        return rescaled_intensity
    
    # phi_0 is not measureable, make a function to fit. (edit: Maybe not necessary)
    def fit_phi_0(self):
        pass
        

    def compute_avg_angular_correlation(self, data, dtype=torch.float64):
        """
        Equation 8: angular cross correlation function
        WARNING this part is computationally expensive, takes 20 seconds on my laptop
        """

        scan_row, scan_col, phi, r = data.shape
        g_all = data.reshape(scan_row * scan_col, phi, r) # Flatten grid of dps to one dimension, makes for loop simpler
        N_alpha = g_all.shape[0] # Number of dps we have = number of configurations (N_alpha)

        dphi = 2.0 * torch.pi / phi # A small change in angle

        C_sum = torch.zeros(phi, r, r)

        for a in range(N_alpha):
            g = g_all[a]
            for d in range(phi):
                shifted = torch.roll(g, shifts=-d, dims=0) # shift by d pixels UP
                C_sum[d] += g.T @ shifted

        C_avg = (C_sum / N_alpha) * dphi
        return C_avg

    def extract_Bl_matrices(self, C_avg, l_max=20, sv_cutoff=0.05):
        # Equation 9, 10, 11
        # for each l:
        #     build the linear system relating C to B_l(q, q')
        #     solve it by SVD
        # Ignore odd l-terms, Friedel symmetry?
        # 5% SVD cutoff
        Nphi, Nq, Nqp = C_avg.shape
        dphi = np.linspace(0, 2 * np.pi, Nphi + 1)[:-1] # Include every dphi using torch.linspace array rather than loop to faciiltate legval
        l_values = list(range(0, l_max, 2)) # This is for the even-only case.
        Nl = len(l_values)
        B_l = torch.zeros((Nl, Nq, Nqp))
        cos_dphi = np.cos(dphi)

        for iq in range(Nq):
            for iqp in range(Nqp): # Handled for each q, qp pair by for loops

                # data vector
                c = C_avg[:, iq, iqp].to(torch.float64)

                # build matrix (including 4pi term here)
                leg_matrix = np.zeros((Nphi, Nl)) # Ndphi by Nl
                for i in range(Nl):
                    coeffs = np.zeros(Nl)
                    coeffs[i] = 1.0
                    leg_matrix[:, i] = legval(cos_dphi, coeffs) / (4 * np.pi)
                leg_matrix = torch.as_tensor(leg_matrix, dtype=torch.float64)

                # SVD solve (from Claude)
                U, S, Vh = torch.linalg.svd(leg_matrix, full_matrices=False)
                S_max = S.max()
                S_inv = torch.where(S > sv_cutoff * S_max, 1.0 / S, torch.zeros_like(S))
                b = Vh.T @ (S_inv * (U.T @ c))

                B_l[:, iq, iqp] = b
        return B_l, l_values

        # TODO: It is mentioned in the paper that a 5% cutoff for SVD is necessary because of "conditioning." When you make a notebook/visuals, demonstrate the need for this by playing with the cutoff.
        # TODO: Compare each function to the martin code to see the difference/similarity or understand the repo better

    def transform_to_real_space(self, Bl_mats, l_values, dq=0.01):
        """
        IN PROGRESS
        dq should be the size of an individual pixel. Default conversion for now is 0.01 angstrom^-1/pixel.
        """
        # Equation 12, 13
        # apply bessel transform twice for each l
        # → shape = (l, r, r')
        
        # Step one is to define q
        q = torch.arange(0, Bl_mats.shape[1]) * dq
        r = torch.reciprocal(q)
        real_Bl = torch.zeros(Bl_mats.shape)

        for l in range(len(l_values)):
            Bl = Bl_mats[l].to(dtype=torch.float64) # The corresponding q x q' matrix
            arg = 2 * torch.pi * torch.outer(r, q) # Shape len(r) x len(q)
            jl = torch.from_numpy(spherical_jn(l_values[l], arg)).to(dtype=torch.float64)

            # Representing DSBT (eq 12) as a transformation matrix "sbessel"
            sbessel = 4 * torch.pi * jl * (q**2) * dq
            # Applying it twice (once along each axis)
            real_Bl[l] = sbessel @ Bl @ sbessel.T * (-1) ** l_values[l]
        
        return real_Bl

    def reconstruct_PADF(self, real_Bl, l_values, Na):
        """
        Reconstruct the pair-angle distribution function from B_l(r, r')
        For each theta:
        - Sum Pl(cos theta) times B_l() over all l
        - Multiply by n_alpha * 2 pi

        NOTE: Theta will go from 0 to pi
        EDIT: Na should be number of atoms not number of dps
        """
        padf = torch.zeros((real_Bl.shape[1], real_Bl.shape[2], 180))
        theta = np.linspace(0, np.pi, 180)
        cos_theta = np.cos(theta)
        chosen_l_values = l_values[1:]

        # Sum over all l
        for l in range(len(chosen_l_values)): # skipping l = 0
            coeffs = np.zeros(chosen_l_values[-1] + 1)
            coeffs[chosen_l_values[l]] = 1
            Pl = torch.tensor(legval(cos_theta, coeffs))
            Bl = real_Bl[l + 1] # skipping l = 0
            padf += Bl[:, :, np.newaxis] * Pl[np.newaxis, np.newaxis, :]

        padf *= 2 * torch.pi * Na
        return padf

    def simple_plot(self):
        """
        Plot the r = r' diagonal of the PADF as a 2D map over (r, theta).

        Uses self.padf, which should be a (Nr, Nr, Ntheta) ndarray
        (from reconstruct_PAD()) defined in __init__.
        """
        # Take padf[i, i, k] for every distance i and angle k -> (Nr, Ntheta)
        diag = np.einsum("iik->ik", self.padf)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.pcolormesh(diag, shading="auto")
        fig.colorbar(im, ax=ax, label=r"$\Theta(r, r, \theta)$")

        ax.set_xlabel(r"$\theta$ index")
        ax.set_ylabel("r = r' index")
        ax.set_title("PADF diagonal")
        fig.tight_layout()

        return ax

