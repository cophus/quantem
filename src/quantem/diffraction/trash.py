def test_extract_Bl_matrices(C_avg, l_max=20, sv_cutoff=0.05):
    Nphi, Nq, Nqp = C_avg.shape
    """EDIT: Dropped the last element from cos_dphi"""
    dphi = np.linspace(0, 2 * np.pi, Nphi + 1)[:-1] # Include every dphi using torch.linspace array rather than loop to faciiltate legval
    l_values = list(range(0, l_max, 2)) # This is for the even-only case. Q: Why even only?
    Nl = len(l_values)
    B_l = torch.zeros(Nl, Nq, Nqp)
    cos_dphi = np.cos(dphi)

    for iq in range(Nq):
        for iqp in range(Nqp): # Handled for each q, qp pair by for loops

            # data vector
            """EDIT: Apparently dtype specification was very important or else you get float * double * float"""
            c = C_avg[:, iq, iqp].to(torch.float64)

            # build legendre matrix (including 4pi term here)
            """EDIT: np.zeros takes in a tuple, not two separate parameters!"""
            leg_matrix = np.zeros((Nphi, Nl)) # Ndphi by Nl
            for i in range(Nl):
                coeffs = np.zeros(Nl)
                coeffs[i] = 1.0
                leg_matrix[:, i] = legval(cos_dphi, coeffs) / (4 * np.pi)
            """EDIT: dtype specification again"""
            leg_matrix = torch.as_tensor(leg_matrix, dtype=torch.float64)

            # SVD solve (from Claude)
            U, S, Vh = torch.linalg.svd(leg_matrix, full_matrices=False)
            S_max = S.max()
            S_inv = torch.where(S > sv_cutoff * S_max, 1.0 / S,
                                torch.zeros_like(S))         # truncate small SVs
            b = Vh.T @ (S_inv * (U.T @ c))                   # least-squares solution

            B_l[:, iq, iqp] = b
    
    return B_l, l_values


def transform_to_real_space(Bl_mats, l_values, dq=0.01):
    """
    IN PROGRESS
    dq should be the size of an individual pixel. Default conversion for now is 0.01 angstrom^-1/pixel.
    """
    # Equation 12, 13
    # for each l:
    #     apply the discrete spherical Bessel transform (DSBT)
    #     once for q→r and once for q'→r'
    #     include the (-1)^l factor from Eq. 13
    # → shape = (l, r, r')
    
    # Step one is to define q
    q = torch.arange(0, Bl_mats.shape[1]) * dq
    r = torch.reciprocal(q)
    real_Bl = torch.zeros(Bl_mats.shape) # EDIT: Should be Bl.shape or Bl_mats.shape???

    """EDIT: It's range(len()) not just len()"""
    for l in range(len(l_values)):
        """EDIT: data type specification again"""
        Bl = Bl_mats[l].to(dtype=torch.float64) # The corresponding q x q' matrix

        # TODO: Understand why size is r, q
        arg = 2 * torch.pi * torch.outer(r, q) # Shape len(r) x len(q)
        """EDIT: data type specification AND numpy to torch conversion"""
        jl = torch.from_numpy(spherical_jn(l_values[l], arg)).to(dtype=torch.float64)

        # Representing DSBT (eq 12) as a transformation matrix "sbessel"
        sbessel = 4 * torch.pi * jl * (q**2) * dq
        real_Bl[l] = sbessel @ Bl @ sbessel.T * (-1) ** l_values[l]
    
    return real_Bl


def reconstruct_PADF(real_Bl, l_values, Na):
    """
    Reconstruct the pair-angle distribution function from B_l(r, r')
    For each theta:
    - Sum Pl(cos theta) times B_l() over all l
    - Multiply by n_alpha * 2 pi

    NOTE: Theta will go from 0 to 180
    """

    """EDIT: Change the order of padf from theta, r, r' to r, r', theta"""
    padf = torch.zeros((real_Bl.shape[1], real_Bl.shape[2], 180))
    theta = np.linspace(0, 179, 180)
    cos_theta = np.cos(theta)
    chosen_l_values = l_values[1:]

    # Sum over all l
    for l in range(len(chosen_l_values)): # skipping l = 0
        """
        EDIT: The coeffs array needs to be as long as the highest l-value, although it might be best to just use
        eval_legendre() from scipy.special because this avoids the coeffs array which is not necessary for our
        purposes.
        """
        coeffs = np.zeros(chosen_l_values[-1] + 1)
        coeffs[chosen_l_values[l]] = 1
        """EDIT: Make Pl a torch tensor"""
        Pl = torch.tensor(legval(cos_theta, coeffs))
        Bl = real_Bl[l]
        padf += Bl[:, :, np.newaxis] * Pl[np.newaxis, np.newaxis, :]
        print(padf.size())

    padf *= 2 * torch.pi * Na
    return padf