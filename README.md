# Tiling optimization case study code

This repository contains example code used in testing and illustrating
development of the tiling optimization algorith and method for
the [GarminCustomMaps](https://github.com/NINAnor/GarminCustomMaps "GarminCustomMaps GitHub repo")
QGIS plugin.

A detailed description of all the code can be found at the
[Geospatial Development Documentation](http://integer0.users.sourceforge.net/tiling-optimization-case-study-1.html)
page.


## Usage
`factors_comparison.py`:

* adjust parameters in the code:

        #Set up sample space
        n = 15
        x = np.logspace(1, n, num=50, base=2, dtype=np.uint32)

* run `python factors_comparison.py`


`optimization.py`:

* adjust parameters in the code:

        # Set up conditions
        x = 6000
        y = 6000
        max_tile_size = 1024 **2
        max_num_tiles = 100
        flag_plot = False
        ofunc = 'dtb'
        savefig = True

* run `python optimization.py`


`ti.*`:

* adjust code in `ti_setup` and `ti_*`
* run with `timeit`:

        $ python -m timeit -n 50 -s "$(<ti_setup)" "$(<ti_code_bool)"
