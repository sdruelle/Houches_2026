#%%
import yt
import numpy as np
import matplotlib.pyplot as plt
from pywavan import fan_trans, nb_scale, powspec
#%%

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",

    "axes.titlecolor": "black",
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
})
#%%
plot_path = "/Xnfs/Houches2026/mberkner/turbulenceanalysis/plots/"
work_path = "/Xnfs/Houches2026/mberkner/turbulenceanalysis/data/"
path_double_B = "/Xnfs/Houches2026/group7/double_B/"
path_default = "/Xnfs/Houches2026/group7/default_setup/"

path_half_B = "/Xnfs/Houches2026/group7/half_B/"
data_sets = {
    "double_B": {
        "path": path_double_B,
        "title": r"$B = 2 \cdot B_0\,\mu\mathrm{G}$",
        "color": "teal",
    },
    "half_B": {
        "path": path_half_B,
        "title": r"$B = 0.5 \cdot B_0\,\mu\mathrm{G}$",
        "color": "darkred",
    },
     "default_setup": {
        "path": path_default,
        "title": r"baseline",
        "color": "yellowgreen",
    },
    "high_turbrms": {
        "path": "/Xnfs/Houches2026/group7/high_turbrms/",
        "title": r"high_turbrms",
        "color": "orange",
    },
    "low_turbrms": {
        "path": "/Xnfs/Houches2026/group7/low_turbrms/",
        "title": r"low_turbrms",
        "color": "blue",
    },
    "mhd_com_forcing": {
        "path": "/Xnfs/Houches2026/group7/mhd_com_forcing/",
        "title": r"mhd_com_forcing",
        "color": "red",
    },
    "mhd_sol_forcing": {
        "path": "/Xnfs/Houches2026/group7/mhd_sol_forcing/",
        "title": r"mhd_sol_forcing",
        "color": "purple",
    },
    "powerlaw_driving": {
        "path": "/Xnfs/Houches2026/group7/powerlaw_driving/",
        "title": r"powerlaw_driving",
        "color": "brown",
    },
    "isothermal": {
        "path": "/Xnfs/Houches2026/group7/run_v2/",
        "title": r"isothermal",
        "color": "pink",
    },
}
iout = 50

#make dir for plots_: 
import os
for key in data_sets.keys():
    if not os.path.exists(plot_path+f"{key}/"):
        os.makedirs(plot_path+f"{key}/")
    if not os.path.exists(work_path+f"{key}/"):
        os.makedirs(work_path+f"{key}/")

for key in data_sets.keys():
    path = data_sets[key]["path"]
    data_sets[key]["data"] = yt.load(path+"output_000{}/info_000{}.txt".format(iout, iout))
#%%
def get_cubes(ds, component="density"):
    levelmin = ds.parameters["levelmin"]
    res = 1 << levelmin 
    grid = ds.r[
        ::res*1j,
        ::res*1j,
    ::res*1j,
    ]
    # Generate projection plot and save result
    if component == "density":
        fits = grid.to_fits_data(
            fields=[("gas", "density"), 
                    ("gas", "pressure"), 
                    ("gas", "velocity_x"), 
                    ("gas", "velocity_y"),
                    ("gas", "velocity_z")], length_unit="pc"
        )
        fits.writeto(f"{work_path}/{key}/{iout}_cube.fits", overwrite=True)


        prj = yt.ProjectionPlot(ds, "z", ("gas", component))
        prj.set_unit(("density"), ("M_sun/pc**2"))

        prj.show()

        # Save it as a FITS file

        prj_fits = yt.FITSProjection(ds, "z", ("gas", "density"), image_res=2**ds.parameters["levelmax"])
        prj_fits.set_unit(("density"), ("M_sun/pc**2"))
        prj_fits.writeto(f"{work_path}/{key}/{iout}_proj_z.fits", overwrite=True)
        plt.imshow(prj_fits[0].data.v)
        plt.colorbar()


        # Run Multi-scale non Gaussian fragmentation
        fit_img = prj_fits[0]
        im = fit_img.data.v
        im_copy = np.copy(im)

    if component=="B_field":
        fits = grid.to_fits_data(
            fields=[("gas", "magnetic_field_x"),
                    ("gas", "magnetic_field_y"),
                    ("gas", "magnetic_field_z")
                    ], length_unit="pc"
        )
        B_cube = np.sqrt(fits["magnetic_field_x"].data**2 + fits["magnetic_field_y"].data**2 + fits["magnetic_field_z"].data**2)

        prj = yt.ProjectionPlot(ds, "z", B_cube)
        prj.set_unit(("B_field"), ("microgauss"))

        prj.show()

        # Save it as a FITS file

        prj_fits = yt.FITSProjection(ds, "z", B_cube, image_res=2**ds.parameters["levelmax"])
        prj_fits.set_unit(("B_field"), ("microgauss"))
        prj_fits.writeto(f"{work_path}/{key}/{iout}_proj_z_Bfield.fits", overwrite=True)
        plt.imshow(prj_fits[0].data.v)
        plt.colorbar()


        # Run Multi-scale non Gaussian fragmentation
        fit_img = prj_fits[0]
        im = fit_img.data.v
        im_copy = np.copy(im)

    q = 2
    wt, S11a, wav_k, S1a, q = fan_trans(im_copy, q=[q]*nb_scale((im.shape[0], im.shape[1])), zeromean=True, nan_frame=True)

    return wt, S11a, wav_k, S1a, q, im
#%%
for key in data_sets.keys():
    wt, S11a, wav_k, S1a, q, im = get_cubes(data_sets[key]["data"])

    data_sets[key]["wt"] = wt   
    data_sets[key]["S11a"] = S11a
    data_sets[key]["wav_k"] = wav_k
    data_sets[key]["S1a"] = S1a
    data_sets[key]["im"]= im
#%%
for key in data_sets.keys():
    wt, S11a, wav_k, S1a, q, im = get_cubes(data_sets[key]["data"], component="B_field")

    data_sets[key]["B_field"]["wt"] = wt   
    data_sets[key]["B_field"]["S11a"] = S11a
    data_sets[key]["B_field"]["wav_k"] = wav_k
    data_sets[key]["B_field"]["S1a"] = S1a
    data_sets[key]["B_field"]["im"]= im
    
#%%
from matplotlib import colors as colorsx

def plot_decomposition(data_sets, key):
    wt = data_sets[key]["wt"]
    S11a = data_sets[key]["S11a"]
    wav_k = data_sets[key]["wav_k"]
    S1a = data_sets[key]["S1a"]
    im = data_sets[key]["im"]
    # Plot the decomposition
    num_scales = len(wav_k)
    meanim = np.nanmean(im)
    coherent = np.sum(wt[num_scales:2*num_scales,:,:].real, axis=0) + meanim
    Gaussian = np.sum(wt[2*num_scales:,:,:].real, axis=0) + meanim


    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11,4), dpi=150)
    cmap='inferno'
    plotmin=1
    plotmax=6
    extent = [-20, 20, -20, 20]
    norm=colorsx.LogNorm(vmin=plotmin, vmax=plotmax)

    imag = axes[0].imshow(im, interpolation='none', cmap=cmap, origin='lower', norm=norm, extent=extent)
    axes[1].imshow( Gaussian, interpolation='none', cmap=cmap, origin='lower', norm=norm, extent=extent)
    axes[2].imshow( coherent, interpolation='none', cmap=cmap, origin='lower', norm=norm, extent=extent)
    axes[0].set(title="Original",  xlabel="x [pc]", ylabel="y [pc]")
    axes[1].set(title="Gaussian",  xlabel="x [pc]", ylabel="y [pc]")
    axes[2].set(title="Coherent",  xlabel="x [pc]", ylabel="y [pc]")
    fig.suptitle("{} decomposition".format(data_sets[key]["title"]))

    cbar = fig.colorbar(imag, ax=axes.ravel().tolist(), orientation="horizontal")
    fig.savefig(plot_path+f"{key}/{iout}_decomposition.png", dpi=300, bbox_inches='tight')
    plt.show()
#%%
# plot_decomposition(data_sets, "double_B")
# plot_decomposition(data_sets, "half_B")
# plot_decomposition(data_sets, "default_setup")
for key in data_sets.keys():
    plot_decomposition(data_sets, key)
#%%
#Plot the segmented power spectra
from matrix_legend import matrix_legend

def plot_power_spectra(data_sets, keys, title):
    fig, ax = plt.subplots(figsize=(6,5))
    for key in keys:
        wt = data_sets[key]["wt"]
        S11a = data_sets[key]["S11a"]
        wav_k = data_sets[key]["wav_k"]
        S1a = data_sets[key]["S1a"]
        # ax.plot(wav_k, S1a[0,:],color=data_sets[key]["color"], label=data_sets[key]["title"] + " | Total", markersize=5, lw=1)

        ax.plot(wav_k, S1a[1,:], 's', linestyle='--', label=data_sets[key]["title"] + " | Coherent",  color=data_sets[key]["color"], markersize=5, lw=2)
        ax.plot(wav_k, S1a[2,:], '^', linestyle=':', label=data_sets[key]["title"] + " | Gaussian", color=data_sets[key]["color"], markersize=5, lw=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('k')
        ax.set_ylabel('P(k)')
        matrix_legend(ax)
    fig.savefig(plot_path+f"segmented_power_spectra_{title}.png", dpi=300, bbox_inches='tight')

plot_power_spectra(data_sets, ["double_B", "half_B", "default_setup"], "Magnetic Field Strength Comparison")
plot_power_spectra(data_sets, ["high_turbrms", "low_turbrms", "default_setup"], "Turbulence Strength Comparison")
plot_power_spectra(data_sets, ["mhd_com_forcing", "mhd_sol_forcing", "default_setup"], "Forcing Comparison")
plot_power_spectra(data_sets, ["powerlaw_driving", "isothermal", "default_setup"], "Power Law Driving Comparison")

# %%
# look at mass distribution of coherent and gaussian components
def plot_mass_distribution(data_sets, keys, title):
    fig, ax = plt.subplots(figsize=(8,5))
    for key in keys:
        wt = data_sets[key]["wt"]
        S11a = data_sets[key]["S11a"]
        wav_k = data_sets[key]["wav_k"]
        S1a = data_sets[key]["S1a"]
        im = data_sets[key]["im"]

        coherent = np.sum(wt[len(wav_k):2*len(wav_k),:,:].real, axis=0) + np.nanmean(im)
        Gaussian = np.sum(wt[2*len(wav_k):,:,:].real, axis=0) + np.nanmean(im)
        # logbins = np.logspace(np.log10(np.nanmin(im)), np.log10(np.nanmax(im)), 50)
        hist, bins, _ = ax.hist(coherent.flatten(), bins=50, density=True, alpha=0.5, label=data_sets[key]["title"] + " | Coherent", color=data_sets[key]["color"])
        ax.hist(Gaussian.flatten(), bins=bins, density=True, alpha=0.5, label=data_sets[key]["title"] + " | Gaussian", color=data_sets[key]["color"], histtype='step', lw=2)
        ax.set_title(title)
        ax.set_xlabel('Density')
        ax.set_ylabel('PDF')
        ax.set_yscale('log')
        # ax.set_xscale('log')
    matrix_legend(ax)
    fig.savefig(plot_path+f"mass_distribution_{title}.png", dpi=300, bbox_inches='tight')
plot_mass_distribution(data_sets, ["double_B", "half_B", "default_setup"], "Magnetic Field Strength Comparison")
plot_mass_distribution(data_sets, ["high_turbrms", "low_turbrms", "default_setup"], "Turbulence Strength Comparison")
plot_mass_distribution(data_sets, ["mhd_com_forcing", "mhd_sol_forcing", "default_setup"], "Forcing Comparison")
plot_mass_distribution(data_sets, ["powerlaw_driving", "isothermal", "default_setup"], "Power Law Driving Comparison")
# compressive more low and high density gas --> see that in the tails of the PDF
# B field: pressure: stabilize small structures and prevent collapse , tension make coherent structure, agline magnetic field 

#%%
# Calculate the Crossings in the powerspectra of Gaussian and Coherent components
def calculate_crossings(data_sets, keys):
    crossings = {}
    for key in keys:
        wt = data_sets[key]["wt"]
        S11a = data_sets[key]["S11a"]
        wav_k = data_sets[key]["wav_k"]
        S1a = data_sets[key]["S1a"]

        coherent = S1a[1,:]
        Gaussian = S1a[2,:]

        crossing_indices = np.where(np.diff(np.sign(coherent - Gaussian)))[0]
        crossing_k_values = wav_k[crossing_indices]
        crossings[key] = crossing_k_values
    return crossings

# Plot them with the keys: 
crossings = calculate_crossings(data_sets, data_sets.keys()) 
fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
ax.grid()

for key in crossings.keys():
    ax.scatter(crossings[key], [key]*len(crossings[key]), label=data_sets[key]["title"], color=data_sets[key]["color"])
ax.set_xlabel('k')
ax.set_ylabel('Simulation')
ax.set_title('Crossings in Power Spectra of \n Gaussian and Coherent Components')
fig.savefig(plot_path+f"crossings_power_spectra.png", dpi=300, bbox_inches='tight')
#Put legend outside of the plot
# %%
# Integrate the power spectra of Gaussian and Coherent components to get the total mass in each component
# Normalize the integrated mass by the total mass in the original image
def integrate_power_spectra(data_sets, keys):
    integrated_masses = {}
    for key in keys:
        wt = data_sets[key]["wt"]
        S11a = data_sets[key]["S11a"]
        wav_k = data_sets[key]["wav_k"]
        S1a = data_sets[key]["S1a"]

        coherent = S1a[1,:]
        Gaussian = S1a[2,:]
        total_mass = np.trapezoid(S1a[0,:], wav_k)

        integrated_coherent_mass = np.trapezoid(coherent, wav_k)
        integrated_Gaussian_mass = np.trapezoid(Gaussian, wav_k)

        integrated_masses[key] = {
            "coherent": integrated_coherent_mass / total_mass if total_mass != 0 else 0,
            "Gaussian": integrated_Gaussian_mass / total_mass if total_mass != 0 else 0
        }
    return integrated_masses
# Plot the integrated masses in a bar plot
integrated_masses = integrate_power_spectra(data_sets, data_sets.keys())
fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
bar_width = 0.35
index = np.arange(len(integrated_masses))
coherent_masses = [integrated_masses[key]["coherent"] for key in integrated_masses.keys()]
Gaussian_masses = [integrated_masses[key]["Gaussian"] for key in integrated_masses.keys()]
ax.bar(index, coherent_masses, bar_width, label='Coherent', color='blue')
ax.bar(index + bar_width, Gaussian_masses, bar_width, label='Gaussian', color='orange')
ax.set_xlabel('Simulation')
ax.set_ylabel('Integrated P(k)')
ax.set_title('Integrated P(k) in Coherent and Gaussian Components')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels([data_sets[key]["title"] for key in integrated_masses.keys()], rotation=45, ha='right')
ax.legend() 
fig.savefig(plot_path+f"integrated_power_spectra.png", dpi=300, bbox_inches='tight')
# %%
