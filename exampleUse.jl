# Example script of using relaxationColor 

using PyPlot
using FileIO
include("RelaxationColor.jl")                   # the location of the julia files from git

fn = "sampleT1map.jld"          # A previously stored file containing the Julia variable "sampleT1map",
                                                #  assumed to be a 2-dimensional (or 3-dimensional) array of Float
im = FileIO.load(fn)["sampleT1map"]

loLev = 400.0; upLev = 2000.0;                  # Example of range to be displayed
imClip, rgb_vec = relaxationColorMap("T1", im, loLev, upLev)  # call to resource, generating a colormap 
cmap = PyPlot.ColorMap("relaxationColor", rgb_vec, length(rgb_vec), 1.0) # translating the colormap to a format digestible by 
                                                                         #  (in this example) PyPlot                    

figure()
imshow(imClip, vmin=loLev, vmax =upLev, interpolation="bicubic", cmap=cmap)
colorbar()
show()

######## Same example with CairoMakie
#=
f=CairoMakie.Figure()
ax=Axis(f[1,1],aspect=DataAspect())
h=heatmap!(ax,rotr90(imClip),colormap=rgb_vec,colorrange=(loLev,upLev))
Colorbar(f[1,2],h)
f
=#