def D_converter(D: float) -> float:
    """
    Convert D from nm^2.s^-1 to µm^2.s^-1
    """
    return D * 1e-6

def C_converter(C: float) -> float:
    """
    Convert C from mol.L^-1 to Fmol.µm^3
    """
    return C

def R_converter(R: float) -> float:
    """
    Convert R from nm to µm
    """
    return R * 1e-3