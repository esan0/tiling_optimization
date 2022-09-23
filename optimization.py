#!/bin/env python
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from timer import Timer
#These are for the factorization algorithm
import math
from itertools import combinations

# Set formal fonts to Computer Modern
# Thanks https://randomwalker.blog/revert-matplotlib-2-0-mathtext-default-font-to-computer-modern/
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


def optimize_bf (x, y, max_tile_size, max_num_tiles):
    '''
    Brute force optimization method
    Calculate optimization score for all combinations in 1<w<x, 1<h<y
    '''

    # Set up solution space
    W, H = np.meshgrid(np.arange(1,x+1), np.arange(1,y+1))

    #First, check if the image extent is smaller than the maximum tile size
    img_ext = x*y
    if max_tile_size >= img_ext:
        Z = np.zeros((y, x))
        Z[-1, -1] = 1
        return (x, y), 0, W, H, Z

    # Tile size
    A = W*H
    # Number of tiles, rows x columns
    N = np.ceil(x/W) * np.ceil(y/H)
    # Pixel remainder shifted by 1 to avoid division by 0
    P = A*N - img_ext + 1

    # Optimization score
    Z = A*(A <= max_tile_size)*(N <= max_num_tiles) * (np.minimum(W,H)/np.maximum(W,H)) / (N*P)

    # Find optimal size
    opt_tile_size = np.unravel_index(np.argmax(Z), Z.shape)

    return (W[opt_tile_size], H[opt_tile_size]), P[opt_tile_size]-1, W, H, Z

def optimize_dtb (x, y, max_tile_size, max_num_tiles):
    '''
    Brute force optimization method improved by reducing memory footprint
    through use of dtypes, as well as reduced processing load through boolean
    indexing of combination conditions.
    '''

    # Set up solution space
    # np.uint32: 32-bit unsigned integer (0 to 4_294_967_295)
    W, H = np.meshgrid(np.arange(1,x+1, dtype=np.uint32), np.arange(1,y+1, dtype=np.uint32))
    Z = np.zeros((y, x), dtype=np.float32)

    #First, check if the image extent is smaller than the maximum tile size
    img_ext = x*y
    if max_tile_size >= img_ext:
        Z[-1, -1] = 1
        return (x, y), 0, W, H, Z

    # Tile size
    A = W*H
    # Calculate the side ratio
    S = np.minimum(W,H)/np.maximum(W,H)
    # Number of tiles, rows x columns
    # Although we're only interested in values <= 500 (max tiles in newer
    # Garmin units), we still need to keep np.uint32 because we don't
    # want the integer to overflow, potentially messing up tile numbers
    N = (np.ceil(x/W) * np.ceil(y/H)).astype(np.uint32)
    # Pixel remainder shifted by 1 to avoid division by 0
    P = A*N - img_ext + 1
    # Mask the values that fall outside the constraints
    m = (A <= max_tile_size) & (N <= max_num_tiles)

    # Optimization score
    Z[m] = A[m] * S[m] / (N[m] * P[m])

    # Find optimal size
    opt_tile_size = np.unravel_index(np.argmax(Z), Z.shape)

    return (W[opt_tile_size], H[opt_tile_size]), P[opt_tile_size]-1, W, H, Z

def trial_division(n):
    '''
    This trial division factorization algorithm is taken from Wikipedia:
    https://en.wikipedia.org/wiki/Trial_division
    Additional lines use the factorization to get all factors of n including
    1 and n.
    '''
    a = []
    while n % 2 == 0:
        a.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            a.append(f)
            n //= f
        else:
            f += 2
    if n != 1: a.append(n)

    b = []
    for i in range(1, len(a)+1):
        b += [math.prod(x) for x in combinations(a,i)]

    b = [x for x in set(b)]
    b.append(1)
    b.sort()

    return b

def optimize_fac (x, y, max_tile_size, max_num_tiles):
    '''
    Factors optimization method for finding solutions with
    perfect coverage.
    '''
    # Get all the factors of x and y
    w = np.array (trial_division (x), dtype=np.uint32)
    h = np.array (trial_division (y), dtype=np.uint32)

    # Set up solution space
    W, H = np.meshgrid (w, h)
    Z = np.zeros_like(W, dtype=np.float32)

    #First, check if the image extent is smaller than the maximum tile size
    img_ext = x*y
    if max_tile_size >= img_ext:
        Z[-1, -1] = 1
        return (x, y), 0, W, H, Z

    # Tile size
    A = W*H
    # Calculate the side ratio
    S = np.minimum(W,H)/np.maximum(W,H)
    # Number of tiles, rows x columns
    # Although we're only interested in values <= 500 (max tiles in newer
    # Garmin units), we still need to keep np.uint32 because we don't
    # want the integer to overflow, potentially messing up tile numbers
    N = (np.ceil(x/W) * np.ceil(y/H)).astype(np.uint32)
    # Pixel remainder is always zero because we're working with factors
    # Mask the values that fall outside the constraints
    m = (A <= max_tile_size) & (N <= max_num_tiles)

    # Optimization score
    Z[m] = A[m] * S[m] / N[m]

    # Check to see if we found anything, set to 1 at (x,y) if not
    if not Z.any(): Z[-1,-1]=1

    # Find optimal size
    opt_tile_size = np.unravel_index(np.argmax(Z), Z.shape)

    return (W[opt_tile_size], H[opt_tile_size]), 0, W, H, Z

def plot_tile (origin, wh, **props):
    vert = [origin, (origin[0], wh[1]), wh, (wh[0], origin[1]), origin]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(vert, codes)
    return patches.PathPatch(path, **props)

def run_optimization (x, y, max_tile_size, max_num_tiles, plot=False, ofunc='bf', savefig=False):
    # Define a dictionary with name to function mapping for optimization function.
    optimization_function = {
            'bf': optimize_bf,
            'dtb': optimize_dtb,
            'fac': optimize_fac}

    # Call optimization function and time it.
    with Timer(name = ofunc, logger=None):
        (a, b), p, W, H, Z = \
            optimization_function[ofunc](x, y, max_tile_size, max_num_tiles)

    if plot:
        # Set up our figure
        fig = plt.figure(figsize=plt.figaspect(0.5), constrained_layout=True)
        fig.suptitle(f'{ofunc} method', fontsize=14, fontweight='bold')
        grid_spec = fig.add_gridspec(2,3)

        # Set up a zero mask
        m = Z>0
        # Get normalized Z score (can ignore minimum value, because it's 0)
        Z_norm = np.zeros_like(Z, dtype=np.float32)
        Z_norm[m] = (Z[m]/Z.max())
        # Get indices of optimal location
        oxy = np.unravel_index(np.argmax(Z), Z.shape)

        # Plot a 3D view of the solution space
        ax = fig.add_subplot(grid_spec[:, 1] , projection='3d')
        # Draw a stem to mark the optimal size location
        ax.stem(W[oxy].ravel(), H[oxy].ravel(), Z_norm[oxy].ravel(),
            linefmt='C7-', markerfmt='none', basefmt='none')
        ax.scatter(W[m], H[m], Z_norm[m], c=Z_norm[m])
        ax.set(xlim=[W[0,0], W[0,-1]])
        ax.set(ylim=[H[0,0], H[-1,0]])
        ax.set(zlim=[0, 1.1])
        ax.set_xlabel ('Tile width')
        ax.set_ylabel ('Tile height')

        # Plot the solution scatter plot
        ax = fig.add_subplot(grid_spec[0,2])
        # We want the larger scores plotted above smaller ones,
        # so we need to sort the coordinates and scores from lowest to highest
        # This is probably not the most elegant way to acheive this goal, but it works!
        Z_sorted = np.argsort(Z_norm[m])
        pc = ax.scatter(W[m][Z_sorted], H[m][Z_sorted],
                    s=(Z_norm[m][Z_sorted]*10)**3,
                    c=Z_norm[m][Z_sorted], alpha=0.65)
        # Draw a dot to mark the optimal size location
        pc = ax.scatter(W[oxy], H[oxy], s= 1, c='gray', marker='.')
        fig.colorbar(pc, ax=ax, label=r'Normalized optimization score $Z$')
        ax.grid(True)
        ax.set(xlim=[W[0,0], W[0,-1]])
        ax.set(ylim=[H[0,0], H[-1,0]])
        ax.set_xlabel ('Tile width')
        ax.set_ylabel ('Tile height')

        # Plot image extent and tile coverage
        ax = fig.add_subplot(grid_spec[1,2])
        image_props = dict(facecolor='yellowgreen', edgecolor='darkolivegreen', lw=3, alpha=.75)
        tile_props = dict(facecolor='none', edgecolor='yellow', lw=0.5, alpha=.75)
        ncols, nrows = -(-x//a), -(-y//b)
        ax.add_patch(plot_tile((1, 1), (x,y), **image_props))
        for i in range (ncols):
            for j in range (nrows):
                patch = plot_tile((i*a+1, j*b+1), ((i+1)*a, (j+1)*b), **tile_props)
                ax.add_patch(patch)

        #Add some space around image
        ax.set_xlim([-0.1*x,1.1*x])
        ax.set_ylim([-0.1*y,1.1*y])
        ax.set(xticks=[1, x, ((i+1)*a)], yticks=[1, y, ((j+1)*b)])

        # Plot optimization parameters
        ax = fig.add_subplot(grid_spec[:,0])
        ax.text(0.1, 0.5,
            f'Image extent:\n{x*y} ({x} x {y})\n' +
            f'Maximum tile size:\n{max_tile_size} ({np.sqrt(max_tile_size):.0f} x {np.sqrt(max_tile_size):.0f})\n\n' +
            r'$A$ = tile size' + '\n' +
            r'$S$ = ratio of tile sides' + '\n' +
            r'$N$ = number of tiles' + '\n' +
            r'$P$ = pixel remainder' + '\n' +
            r'Optimization score: $Z=\frac{AS}{NP}$' + '\n' +
            f'Optimization fuction: {ofunc}' + '\n\n' +
            f'Calculation time: {Timer.timers[ofunc]:.4} s\n\n' +
            #f'Draw time: {Timer.timers["plotter"]:.4} s' + '\n\n' +
            f'Optimal tile size: {a} x {b}' + '\n' +
            f'Pixel remainder: {p}',
            horizontalalignment='left', verticalalignment='center'
            )
        ax.set(xticks=[], yticks=[])
        ax.set_facecolor('lavenderblush')

        # Show or plot everything
        if savefig:
            fig.savefig(f'./figs/method_{ofunc}_{x}-{y}.png')
        else:
            plt.show()

    else:
            print (
            f'Image extent: {x*y} ({x} x {y})\n' +
            f'Maximum tile size: {max_tile_size}\n\n' +
            'A = tile size' + '\n' +
            'S = ratio of tile sides (always < 1)' + '\n' +
            r'N = number of tiles' + '\n' +
            r'P = pixel remainder' + '\n' +
            r'Optimization score: Z = AS/NP' + '\n' +
            f'Optimization fuction: {ofunc}' + '\n\n' +
            f'Calculation time: {Timer.timers[ofunc]:.4} s\n\n' +
            f'Optimal tile size: {a} x {b}' + '\n' +
            f'Pixel remainder: {p}',
            )

if __name__ == '__main__':
    # Set up conditions
    x = 6000
    y = 6000
    max_tile_size = 1024 **2
    max_num_tiles = 100
    flag_plot = False
    ofunc = 'dtb'
    savefig = True

    run_optimization (x=x, y=y,
        max_tile_size=max_tile_size, max_num_tiles=max_num_tiles,
        plot=flag_plot, ofunc=ofunc, savefig=savefig)
