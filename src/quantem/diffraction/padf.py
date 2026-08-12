import matplotlib.pyplot as plt
import torch
import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_legendre, spherical_jn
from numpy.polynomial.legendre import legval

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.io.serialize import load
from quantem.diffraction.polar_transform import (
    as_dataset4dstem,
    find_origin_angular_grid,
    polar_transform,
)

class PairAngleDistributionFunction(AutoSerialize):
    """
    Compute the pair-angle distribution function from diffraction data:
    a single 2D pattern (Dataset2d), a 3D stack of patterns, or a 4D-STEM
    dataset. Non-4D inputs are wrapped as (1, 1, H, W) / (N, 1, H, W).

    Run the following pipeline:
    padf = PairAngleDistributionFunction(ds)
    1. find_origin_angular_grid(ds)
    2. polar_transform(ds, origin_array=self.origin)
    3. rescale_intensity(self.polar, rho=1, fq=1)
    4. compute_avg_angular_correlation(self.polar_rescaled)
    5. correct_angular_correlation(self.ang_corr)  # theta-mean subtraction + |sin(theta)| factor
    6. extract_Bl_matrices(self.ang_corr_corrected)
    7. transform_to_real_space(self.Bl_mats[0], self.Bl_mats[1])
    8. reconstruct_PADF(self.real_Bl_mats, self.Bl_mats[1], self.ds.shape[0] * self.ds.shape[1])

    """
    def __init__(self,
                 ds=None,
                 origin: NDArray | None = None,
                 polar_ds: Polar4dstem | None = None,
                 ang_corr: torch.Tensor | None = None,
                 Bl_mats = None
                 ):
        super().__init__()

        # accept a single 2D pattern, a 3D stack, or a 4D-STEM dataset
        self.ds = as_dataset4dstem(ds) if ds is not None else None
        self.origin = origin if origin is not None else find_origin_angular_grid(ds)
        self.polar = polar_ds if polar_ds is not None else polar_transform(ds, origin_array=self.origin)

        # TODO: we get rho and fq from sample identity, so either the user provides them, we find them in the dataset, or we look them up based on sample identity
        self.polar_rescaled = self.rescale_intensity(self.polar, rho=1, fq=1)

        self.ang_corr = ang_corr if ang_corr is not None else self.compute_avg_angular_correlation(self.polar_rescaled)
        self.ang_corr_corrected = self.correct_angular_correlation(self.ang_corr)
        self.Bl_mats = Bl_mats if Bl_mats is not None else self.extract_Bl_matrices(self.ang_corr_corrected)
        # radial q calibration of the polar dataset: bin i sits at q_min + i * dq
        dq = float(np.asarray(self.polar.sampling)[3])
        q_min = float(np.asarray(self.polar.origin)[3])
        self.real_Bl_mats = self.transform_to_real_space(self.Bl_mats[0], self.Bl_mats[1], dq=dq, q_min=q_min)
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

        Handles dtype conversion to float64
        """
        
        # TODO: Look at the Martin code to see how he calculated atoms in beam and phi_0
        intensity = data.tensor.to(dtype=torch.float64)
        rescaled_intensity = intensity / (rho * fq**2) # / still need to include atoms in beam and phi
        return rescaled_intensity
    
    # phi_0 is not measureable, make a function to fit. (edit: Maybe not necessary)
    def fit_phi_0(self):
        pass
        

    def compute_avg_angular_correlation(self, data):
        """
        Equation 8: angular cross correlation implemented via Fourier correlation theorem
        """

        scan_row, scan_col, phi, r = data.shape
        g_all = data.reshape(scan_row * scan_col, phi, r) # Flatten grid of dps to one dimension, makes for loop simpler
        N_alpha = g_all.shape[0] # Number of dps we have = number of configurations (N_alpha)

        dphi = 2.0 * torch.pi / phi # A small change in angle

        G = torch.fft.rfft(g_all, dim=1) # Fourier transform along phi axis
        P = torch.einsum('afi,afj->fij', G.conj(), G) # Outer product of r between G and its conjugate, summed over all N_alpha
        C_sum = torch.fft.irfft(P, n=phi, dim=0)

        C_avg = (C_sum / N_alpha) * dphi
        return C_avg

    def correct_angular_correlation(self,
                                    C_avg: torch.Tensor,
                                    subtract_mean: bool = True,
                                    sintheta: bool = True):
        """
        Corrections applied to the angular correlation before B_l extraction,
        matching the pypadf maskcorr.py defaults:

        - subtract_mean: subtract the theta-average from each (q, q') ring,
          removing the dominant isotropic (uncorrelated) background.
        - sintheta: multiply by |sin(theta)| (evaluated at bin centers).
          This is the conversion factor between the correlation function and
          the PADF (eq 11), applied in q-space as in the pypadf workflow.
        """
        C = C_avg.clone()
        Nphi = C.shape[0]
        if subtract_mean:
            C -= C.mean(dim=0, keepdim=True)
        if sintheta:
            theta = 2.0 * torch.pi * (torch.arange(Nphi, dtype=C.dtype) + 0.5) / Nphi
            C *= torch.abs(torch.sin(theta))[:, None, None]
        return C

    def extract_Bl_matrices(self, C_avg, l_max=40, sv_cutoff=0.05):
        # Equation 9, 10, 11
        # for each l:
        #     build the linear system relating C to B_l(q, q')
        #     solve it by SVD
        # Ignore odd l-terms, Friedel symmetry?
        # 5% SVD cutoff
        Nphi, Nq, Nqp = C_avg.shape
        dphi = torch.linspace(0, 2 * torch.pi, Nphi + 1, dtype=torch.float64)[:-1] # Ranges from [0, 2pi)
        l_values = torch.arange(0, l_max, 2) # Even only
        Nl = len(l_values)
        cos_dphi = torch.cos(dphi)

        C_flat = C_avg.reshape(Nphi, Nq * Nqp)

        # Build matrix
        L = l_values.reshape(1, Nl)
        X = cos_dphi.reshape(Nphi, 1)
        leg_matrix = eval_legendre(L, X) # Shape due to broadcasting (Nphi, Nl)

        U, S, Vh = torch.linalg.svd(leg_matrix, full_matrices=False)
        S_max = S.max()
        S_inv = torch.where(S > sv_cutoff * S_max, 1.0 / S, torch.zeros_like(S))

        tmp = U.T @ C_flat
        tmp = S_inv.unsqueeze(1) * tmp
        B_flat = Vh.T @ tmp

        return B_flat.reshape(Nl, Nq, Nqp), l_values

        # TODO: Compare each function to the martin code to see the difference/similarity or understand the repo better

    def transform_to_real_space(self, Bl_mats, l_values, dq=0.01, q_min=0.0,
                                r_min=0.0, r_max=20.0, r_step=0.02):
        """
        Transform B_l(q, q') to B_l(r, r') by applying the spherical Bessel
        transform (eq 8) along each q axis via direct quadrature.

        dq : radial q step of the polar dataset (e.g. angstrom^-1 / bin).
        q_min : q value of the first radial bin (nonzero if the polar
            transform used radial_min > 0).
        r_min, r_max, r_step : real-space grid, in the reciprocal units of q.
        """
        # Equation 12, 13
        # apply bessel transform twice for each l
        # → shape = (l, r, r')

        # Step one is to define q
        q = q_min + torch.arange(0, Bl_mats.shape[1], dtype=torch.float64) * dq
        r = torch.arange(r_min, r_max, r_step, dtype=torch.float64)
        self.r = r.numpy()
        real_Bl = torch.zeros((Bl_mats.shape[0], r.shape[0], r.shape[0]), dtype=torch.float64)

        for l in range(len(l_values)):
            Bl = Bl_mats[l].to(dtype=torch.float64) # The corresponding q x q' matrix
            arg = 2 * torch.pi * torch.outer(r, q) # Shape len(r) x len(q)
            jl = torch.from_numpy(spherical_jn(l_values[l], arg)).to(dtype=torch.float64)

            # Representing DSBT (eq 12) as a transformation matrix "sbessel"
            sbessel = 4 * torch.pi * jl * (q**2) * dq
            # Applying it twice (once along each axis)
            real_Bl[l] = sbessel @ Bl @ sbessel.T * (-1)**l_values[l]
        
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
        padf = torch.zeros((real_Bl.shape[1], real_Bl.shape[2], 180), dtype=torch.float64)
        theta = np.linspace(0, np.pi, 180)
        self.theta = theta
        self.theta_deg = np.degrees(theta)
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
            Plot the r = r' diagonal of the PADF as a 2D map with
            r = r' on the x-axis and theta on the y-axis.

            Uses self.padf, which should be a (Nr, Nr, Ntheta) ndarray
            (from reconstruct_PAD()) defined in __init__.
            """
            # Take padf[i, i, k] for every distance i and angle k -> (Nr, Ntheta)
            diag = np.einsum("iik->ik", self.padf)

            # Transpose so rows = theta, columns = r  -> (Ntheta, Nr)
            diag = diag.T

            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.pcolormesh(diag, shading="auto")
            fig.colorbar(im, ax=ax, label=r"$\Theta(r, r, \theta)$")

            ax.set_xlabel("r = r' index")
            ax.set_ylabel(r"$\theta$ index")
            ax.set_title("PADF diagonal")
            fig.tight_layout()

            return ax

