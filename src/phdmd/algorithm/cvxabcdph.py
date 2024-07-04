
from phdmd.utils.preprocess_data import get_Riccati_transform
from phdmd.algorithm.cvxabcd import cvxabcd

def cvxabcdph(X,
    Y,
    U,
    dXdt=None,
    delta_t=None,
    delta=1e-12,
    constraint_type="no",  # "KYP" | "no" | "nsd" | "nsd"):
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

    # transform to ph-system
    T, phlti_model = get_Riccati_transform(lti_model, get_phlti=True)

    if phlti_model:
        return phlti_model
    else:
        return lti_model