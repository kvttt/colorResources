import matplotlib.colors as mcolors
import numpy as np
from typing import Tuple


def relaxationColorMap(maptype: str, x: np.ndarray, loLev: float, upLev: float) -> Tuple[np.ndarray, mcolors.ListedColormap]:
    """
    relaxationColorMap acts in two ways:
        1. Generate a colormap to be used on display, given image type 
           (which must be one of 
           "T1","R1","T2","T2*","R2","R2*","T1rho","T1ρ","R1rho","R1ρ",
           "t1","r1","t2","t2*","r2","r2*","t1rho","t1ρ","r1rho","r1ρ")
           and given the range of the image to be displayed;
        2. Generates a 'clipped' image, which is a copy of the input image except that values are clipped to the lower level,
           while respecting the special value of 0 (which has to map to the "invalid" color)

    Parameters
    ----------
    maptype : str
        A string from aformentioned series, e.g., "T1" or "R2"
    x : np.ndarray
        The image to be displayed
    loLev : float
        Lower level of the range to be displayed
    upLev: float
        Upper level of the range to be displayed
    
    Returns
    -------
    xClip : np.ndarray
        Value-clipped image
    cmap : mcolors.ListedColormap
        Colormap to be used in image-display functions
    
    """
    
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
        xClip = np.where(
            x < eps, 
            loLev - eps, 
            x,
        )
    else:
        xClip = np.where(
            x < eps, 
            loLev - eps, 
            np.where(x < loLev + eps, loLev + 1.5 * eps, x),
        )

    lutCmap = colorLogRemap(colortable, loLev, upLev)
    
    rgb_vec = lutCmap[:, :3]
    cmap = mcolors.ListedColormap(rgb_vec)
    
    return xClip, cmap


def colorLogRemap(oriCmap: np.ndarray, loLev: float, upLev: float) -> np.ndarray:
    """
    colorLogRemap: 
        Lookup of the original color map table according to a "log-like" curve.
        The log-like curve contains a linear part and a logarithmic part; 
        the size of the parts depends on the range (loLev, upLev)

    Parameters
    ----------
    oriCmap : np.ndarray
        Original color map table
    loLev : float
        Lower level of the range to be displayed
    upLev : float
        Upper level of the range to be displayed
    
    Returns
    -------
    logCmap : np.ndarray
        Modified colormap

    """

    if not upLev > 0:
        raise ValueError("Upper level must be positive")
    if not upLev > loLev:
        raise ValueError("Upper level must be larger than lower level")
    
    mapLength = oriCmap.shape[0]
    eInv = np.exp(-1.0)
    aVal = eInv * upLev
    mVal = max(aVal, loLev)
    bVal = 1.0 / mapLength if aVal < loLev else (aVal - loLev) / (2 * aVal - loLev) + (1.0 / mapLength)
    bVal += 1e-7
    logCmap = np.zeros_like(oriCmap)
    logCmap[0, :] = oriCmap[0, :]
    logPortion = 1.0 / (np.log(mVal) - np.log(upLev))
    
    for g in range(1, mapLength):
        f = 0.0
        x = (g + 1) * (upLev - loLev) / mapLength + loLev
        if x > mVal:
            f = mapLength * ((np.log(mVal) - np.log(x)) * logPortion * (1 - bVal) + bVal)
        elif loLev < aVal and x > loLev:
            f = mapLength * ((x - loLev) / (aVal - loLev) * (bVal - (1.0 / mapLength))) + 1.0
        elif x <= loLev:
            f = 1.0

        logCmap[g, :] = oriCmap[min(mapLength, 1 + int(np.floor(f))) - 1, :]

    return logCmap
