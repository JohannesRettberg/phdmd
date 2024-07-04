def getsystem(PHform):
    # % From DH-form to system form
    # %
    # % This is useful function to transform a solution of stablePassiveFGM
    # % to the system

    sys = {}
    sys["A"] = (PHform["J"] - PHform["R"]) @ (PHform["Q"])
    sys["E"] = PHform["M"]
    sys["B"] = PHform["F"] - PHform["P"]
    sys["C"] = (PHform["F"] + PHform["P"]).T @ (PHform["Q"])
    sys["D"] = PHform["S"] + PHform["N"]
    return sys
