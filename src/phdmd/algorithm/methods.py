from phdmd.algorithm.dmd import iodmd, operator_inference
from phdmd.algorithm.phdmd import phdmd
from phdmd.algorithm.phdmdh import phdmdh
from phdmd.algorithm.cvxabcdph import cvxabcdph
from phdmd.algorithm.cvxabcdpr import cvxabcdpr
from phdmd.algorithm.cvxph import cvxph
from phdmd.utils.system import to_lti, to_phlti

from matplotlib import pyplot as plt


class Method:
    """
    Base class for all methods.
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, X, Y, U, delta_t, H):
        pass


class IODMDMethod(Method):
    def __init__(self):
        super().__init__("DMD")

    def __call__(self, X, Y, U, delta_t, H):
        n = X.shape[0]
        AA_dmd, e = iodmd(X, Y, U)
        lti_dmd = to_lti(AA_dmd, n=n, no_feedtrough=True, sampling_time=delta_t)
        return lti_dmd


class OIMethod(Method):
    def __init__(self):
        super().__init__("OI")

    def __call__(self, X, Y, U, delta_t, H):
        n = X.shape[0]
        AA_dmd, e = operator_inference(X, Y, U, delta_t, H)
        lti_dmd = to_lti(AA_dmd, n=n, E=H, no_feedtrough=True)

        return lti_dmd


class PHDMDMethod(Method):
    def __init__(self):
        super().__init__("pHDMD")

    def __call__(
        self,
        X,
        Y,
        U,
        delta_t,
        H=None,
        dXdt=None,
        use_Berlin=None,
        Q=None,
        use_cvx=False,
        J_known=None,
    ):
        n = X.shape[0]
        J, R, e = phdmd(
            X, Y, U, delta_t=delta_t, H=H, dXdt=dXdt, use_cvx=use_cvx, J_known=J_known
        )
        phlti = to_phlti(J, R, n=n, E=H, no_feedtrough=True)

        return phlti


class PHDMDHMethod(Method):
    def __init__(self):
        super().__init__("pHDMDH")

    def __call__(
        self,
        X,
        Y,
        U,
        delta_t,
        use_Berlin,
        H=None,
        Q=None,
        use_cvx=False,
        J_known=None,
        **kwargs,
    ):
        # H not needed but used to be in accordance with other methods calling
        n = X.shape[0]
        max_iter = 10
        if "ordering" in kwargs:
            ordering = kwargs["ordering"]
        else:
            ordering = "JRH"

        J, R, H, Q, e = phdmdh(
            X,
            Y,
            U,
            H0=H,
            Q0=Q,
            delta_t=delta_t,
            use_Berlin=use_Berlin,
            max_iter=max_iter,
            use_cvx=use_cvx,
            J_known=J_known,
            ordering=ordering,
        )
        phlti = to_phlti(J, R, n=n, E=H, Q=Q, no_feedtrough=True)

        plot_error_over_iter = True
        if plot_error_over_iter:
            plt.figure()
            plt.plot(e, label="rel. Fro error")
            plt.yscale("log")
            plt.xlabel("Iterations")
            plt.ylabel("Rel. Fro norm error")
            plt.savefig(f"error_fro_Bform{use_Berlin}_iter{max_iter}.png")

        return phlti


class CVXABCDPHMethod(Method):
    def __init__(self):
        super().__init__("CVXABCD")

    def __call__(
        self,
        X,
        Y,
        U,
        delta_t,
        use_Berlin,
        H=None,
        Q=None,
        use_cvx=False,
        J_known=None,
        constraint_type="no",
    ):
        max_iter = None
        phlti = cvxabcdph(
            X,
            Y,
            U,
            # dXdt=dXdt,
            delta_t=delta_t,
            # delta=delta,
            constraint_type=constraint_type,  # "KYP" | "no" | "nsd" | "nsd"):
        )

        return phlti


class CVXABCDPRMethod(Method):
    def __init__(self):
        super().__init__("CVXABCDPR")

    def __call__(
        self,
        X,
        Y,
        U,
        delta_t,
        use_Berlin,
        H=None,
        Q=None,
        use_cvx=False,
        J_known=None,
        constraint_type="no",
        **kwargs,
    ):

        if "gillis_options" in kwargs:
            gillis_options = kwargs["gillis_options"]
        else:
            gillis_options = None
        phlti = cvxabcdpr(
            X,
            Y,
            U,
            # dXdt=dXdt,
            delta_t=delta_t,
            # delta=delta,
            constraint_type=constraint_type,  # "KYP" | "no" | "nsd" | "nsd"):
            gillis_options=gillis_options,
        )

        return phlti


class CVXPHMethod(Method):
    def __init__(self):
        super().__init__("CVXPH")

    def __call__(
        self, X, Y, U, delta_t, use_Berlin, H=None, Q=None, use_cvx=False, J_known=None
    ):
        max_iter = None
        phlti = cvxph(
            X,
            Y,
            U,
            H0=H,
            Q0=Q,
            delta_t=delta_t,
            use_Berlin=use_Berlin,
            max_iter=max_iter,
            use_cvx=use_cvx,
            J_known=J_known,
        )

        return phlti


# methods = [IODMDMethod(), OIMethod(), PHDMDMethod()]
