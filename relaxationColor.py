import matplotlib.colors as mcolors
import numpy as np
from typing import Tuple


def relaxationColorMap(maptype: str, x: np.ndarray, loLev: float, upLev: float) -> Tuple[np.ndarray, mcolors.ListedColormap]:
    maptype = maptype.capitalize()
    if maptype in ["T1", "R1"]:
        fn = "./lipari.csv"
    elif maptype in ["T2", "T2*", "R2", "R2*", "T1rho", "T1ρ", "R1rho", "R1ρ"]:
        fn = "./navia.csv"
    else:
        raise ValueError("Expect 'T1', 'T2', 'R1', or 'R2' as maptype")

    colortable = np.genfromtxt(fn, delimiter=' ')
    
    if maptype[0] == 'R':
        colortable = np.flipud(colortable)

    colortable[0, :] = 0.0

    eps = (upLev - loLev) / colortable.shape[0]
    xClip = np.where(x < eps, loLev - eps, np.where(x < loLev + eps, loLev + 1.5 * eps, x))
    
    if loLev < 0:
        xClip = np.where(x < eps, loLev - eps, x)

    lutCmap = colorLogRemap(colortable, loLev, upLev)
    
    rgb_vec = lutCmap[:, :3]
    cmap = mcolors.ListedColormap(rgb_vec)
    
    return xClip, cmap


def colorLogRemap(oriCmap: np.ndarray, loLev: float, upLev: float) -> np.ndarray:
    if not upLev > 0:
        raise ValueError("Upper level must be positive")
    if not upLev > loLev:
        raise ValueError("Upper level must be larger than lower level")
    
    mapLength = oriCmap.shape[0]
    logCmap = np.zeros_like(oriCmap)
    eInv = np.exp(-1.0)
    aVal = eInv * upLev
    mVal = max(aVal, loLev)
    bVal = (aVal < loLev) and (1.0 / mapLength) or (aVal - loLev) / (2 * aVal - loLev) + (1.0 / mapLength)
    bVal += 1e-7

    logCmap[0, :] = oriCmap[0, :]

    logPortion = 1.0 / (np.log(mVal) - np.log(upLev))
    
    for g in range(1, mapLength):
        x = g * (upLev - loLev) / mapLength + loLev
        if x > mVal:
            f = mapLength * ((np.log(mVal) - np.log(x)) * logPortion * (1 - bVal) + bVal)
        elif loLev < aVal and x > loLev:
            f = mapLength * ((x - loLev) / (aVal - loLev) * (bVal - (1.0 / mapLength))) + 1.0
        elif x <= loLev:
            f = 1.0

        logCmap[g, :] = oriCmap[min(mapLength - 1, int(np.floor(f))), :]

    return logCmap
