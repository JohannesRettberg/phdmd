from phdmd.algorithm.cvxabcd import cvxabcd
from phdmd.algorithm.stablePassiveFGM import stablePassiveFGM
from phdmd.utils.gillis_options import GillisOptions

def cvxabcdpr(X,
    Y,
    U,
    dXdt=None,
    delta_t=None,
    delta=1e-12,
    constraint_type="no",  # "KYP" | "no" | "nsd" | "nsd"):
    gillis_options = None,
    ):

    lti_model = cvxabcd(
    X,
    Y,
    U,
    dXdt=dXdt,
    delta_t=delta_t,
    delta=delta,
    constraint_type=constraint_type,  # "KYP" | "no" | "nsd" | "nsd"
    )

    A,B,C,D,E = lti_model.to_abcde_matrices()

    sys = {}
    sys["A"] = A
    sys["B"] = B
    sys["C"] = C
    sys["D"] = D
    sys["E"] = E

    assert isinstance(gillis_options,GillisOptions) 
    gillis_options = gillis_options.get_options()

    phlti_model, eg, tg = stablePassiveFGM(sys, gillis_options)

    return phlti_model

