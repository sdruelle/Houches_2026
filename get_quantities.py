import yt
import os  

yt.enable_parallelism()

ts = yt.load("/Xnfs/Houches2026/sdruelle/sims/double_B/output*")
N  = 2**7

for ds in ts.piter():
    levelmin = ds.parameters["levelmin"]
    res = 1 << levelmin 
    grid = ds.r[
        ::res*1j,
        ::res*1j,
        ::res*1j,
    ]
    fits = grid.to_fits_data(
        fields=[("gas", "density"), 
                ("gas", "pressure"), 
                ("gas", "velocity_x"), 
                ("gas", "velocity_y"),
                ("gas", "velocity_z")], 
        length_unit="pc"
    )
    
    clean_basename = os.path.splitext(ds.basename)[0]
    
    output_filename = f"cube_{clean_basename}.fits"
    
    fits.writeto(output_filename, overwrite=True)