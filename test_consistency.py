from juliacall import Main as jl
import matlab.engine
import numpy as np
from relaxationColor import relaxationColorMap


if __name__ == "__main__":
    # config
    maptype = "T1"
    im = np.load('sampleT1map.npy')
    loLev = 400.0
    upLev = 2000.0

    # Julia 
    jl.include("relaxationColor.jl")
    imClip_julia, cmap_julia = jl.relaxationColorMap(maptype, im, loLev, upLev)
    imClip_julia = np.asarray(imClip_julia)
    cmap_julia = np.asarray([[x.r, x.g, x.b] for x in cmap_julia])

    # Python
    imClip_python, cmap_python = relaxationColorMap(maptype, im, loLev, upLev)
    imClip_python = np.asarray(imClip_python)
    cmap_python = cmap_python.colors

    # MATLAB
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath("./"))
    imClip_matlab, cmap_matlab = eng.relaxationColorMap(maptype, im, loLev, upLev, nargout=2)
    imClip_matlab = np.asarray(imClip_matlab)
    cmap_matlab = np.asarray(cmap_matlab)

    if not np.allclose(imClip_julia, imClip_python):
        print("imClip mismatch between Julia and Python")
    if not np.allclose(cmap_julia, cmap_python):
        print("cmap mismatch between Julia and Python")
    if not np.allclose(imClip_julia, imClip_matlab):
        print("imClip mismatch between Julia and MATLAB")
    if not np.allclose(cmap_julia, cmap_matlab):
        print("cmap mismatch between Julia and MATLAB")
