from astropy.io import fits
import os
import subprocess
import numpy as np

def apogee_filename(res, apid):
    return f"aspcapStar-r8-{res}-{apid}.fits"

def download_apogee_spectra(df, directory="apogee_spectra"):
    nspectra = len(df)
    filenames = [apogee_filename(row["RESULTS_VERSION"], row["APOGEE_ID"]) for row in df]
    downloaded = set(os.listdir(directory))

    s = len(set(filenames).intersection(downloaded))
    print(f"of {nspectra} spectra, {nspectra - s} need to be downloaded")

    for row in df:
        filename = apogee_filename(row["RESULTS_VERSION"], row["APOGEE_ID"]) 
        if filename not in downloaded:
            url = "https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/" + \
                   f"stars/{row['ASPCAP_VERSION']}/{row['RESULTS_VERSION']}/" + \
                   f"{row['LOCATION_ID']}/{filename}"
            subprocess.run(["wget", "-nv", "-P", directory, url]) 

def load_apogee_spectrum(row, directory="apogee_spectra"):
    with fits.open(directory + '/' + \
            apogee_filename(row["RESULTS_VERSION"], row["APOGEE_ID"])) as hdus:
            flux = hdus[1].data
            err  = hdus[2].data
    return flux, err

def lamost_filename(obsid):
    return f"{obsid}?token="

def download_lamost_spectra(obsids, directory="lamost_spectra"):
    nspectra = len(obsids)
    filenames = set([lamost_filename(oid) for oid in obsids])
    downloaded = set(os.listdir(directory))
    s = len(filenames.intersection(downloaded))
    print(f"of {nspectra} spectra, {nspectra - s} need to be downloaded")
    for fn in filenames - downloaded:
        url = f"http://dr4.lamost.org/./spectrum/fits/{fn}"
        subprocess.run(["wget", "-P", directory, url])


def smoothing_kernel(wl, wl0, L):
    return np.exp(-1*np.square((wl-wl0)/L) / 2)

def calculate_continuum(wls, fluxes, ivars, L=100, window=300):
    for (wl, flux, ivar) in zip(wls, fluxes, ivars):
        indices = np.logical_and(wl-window < wls, wls < wl+window)
        weights = smoothing_kernel(wls[indices], wl, L) * ivars[indices]
        yield np.sum(weights * fluxes[indices])/np.sum(weights)

def load_lamost_spectrum(obsid, wl_grid=np.load("wl_grid.npy"), 
                         directory="lamost_spectra", 
                         continuum_normalize=True):
    with fits.open(directory + '/' + lamost_filename(obsid)) as hdus:
        hdu = hdus[0]
        wl = hdu.data[2] - (hdu.data[2] * hdu.header["Z"])
        flux = hdu.data[0]
        ivar = hdu.data[1]

    if continuum_normalize:
        cont = np.array(list(calculate_continuum(wl, flux, ivar)))
        flux /= cont
        ivar *= np.square(cont)

    if wl_grid is not None:
        flux = np.interp(wl_grid, wl, flux)
        ivar = np.interp(wl_grid, wl, ivar)
        wl = wl_grid
    return wl, flux, ivar



