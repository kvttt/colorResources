import matplotlib.pyplot as plt
import numpy as np
from relaxationColor import relaxationColorMap


if __name__ == "__main__":
    fn = 'sampleT1map.npy'
    im = np.load(fn)

    loLev = 400.0
    upLev = 2000.0

    imClip, cmap = relaxationColorMap("T1", im, loLev, upLev)
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.imshow(imClip, vmin=loLev, vmax=upLev, cmap=cmap)
    plt.colorbar()
    plt.show()
