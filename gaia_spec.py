"""Wrapper around GaiaXPy functions for convenience and better compatibility with QUBRICS data."""

import numpy as np
import matplotlib.pylab as plt
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
import os
import gaiaxpy
from tqdm.auto import tqdm

# ---------------------------------------------------------------------- #
# Needed for Marz Conversion


def generateComment(DBData):
    """
    Generates the comment string in the Fibres Extension.
    """
    t = DBData[3]
    z = DBData[4]
    tf = "P" if DBData[5] == "" else DBData[5]
    qf = DBData[6]
    n = DBData[7]
    return str(t) + " " + str(z) + " " + tf + qf + " - " + n


def isNumber(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def writeFits(flux, error, wave, fibre=None, name="MarzConverterOutput.fits"):
    """
    Writes the process fits file, ready for Marz. Asks for overwrite permission!
    """
    primaryHDU = fits.PrimaryHDU(flux)
    varianceHDU = fits.ImageHDU(error, name="variance")
    waveHDU = fits.ImageHDU(wave, name="wavelength")

    if fibre is None:
        hduListOut = fits.HDUList([primaryHDU, varianceHDU, waveHDU])
    else:
        hduListOut = fits.HDUList([primaryHDU, varianceHDU, waveHDU, fibre])

    try:
        hduListOut.writeto(name)
        hduListOut.close()
    except OSError:
        overwrite = input("File already exists, overwrite (Y/n)? ")
        if overwrite.lower() == "y" or overwrite == "":
            hduListOut.writeto(name, overwrite=True)
            hduListOut.close()
        else:
            hduListOut.close()


def _getObservationData(name):
    return [name, "0", "0", "-", "-", "-", "-", "-"]


def getObservationData(nameList):
    """
    Generates mock data if everything fails.
    """
    observationDataFallback = []
    if not isinstance(nameList, list):
        nameList = [nameList]

    for name in nameList:
        mockData = _getObservationData(name)
        observationDataFallback.append(mockData)
    return np.array(observationDataFallback)


def generateFibresData(data):
    """
    Given data from QDB, produces fibre data.
    If no data are found on the DB (or the DB can't be accessed) neutral, mock
    data are generated on the fly (e.g. ra/dec = 0/0, `z_spec = -`)
    """
    name, t, ra, dec, comm = [], [], [], [], []
    t = ["P"] * len(data)

    for data in data:
        z_name = float(data[4]) if isNumber(data[4]) else -1
        name.append(str(data[0]) + " - " + str(round(z_name, 2)))
        ra.append(str(float(data[1]) * np.pi / 180))
        dec.append(str(float(data[2]) * np.pi / 180))
        comm.append(generateComment(data))

    nameCol = fits.Column(name="NAME", format="80A", array=name)
    typeCol = fits.Column(name="TYPE", format="1A", array=t)
    raCol = fits.Column(name="RA", format="1D", array=ra)
    decCol = fits.Column(name="DEC", format="1D", array=dec)
    commCol = fits.Column(name="COMMENT", format="80A", array=comm)

    outCols = fits.ColDefs([nameCol, typeCol, raCol, decCol, commCol])
    return fits.BinTableHDU().from_columns(outCols, name="fibres")


def writeAll(specList, outpath, outfile, wmin = 3500 *u.AA, wmax = 10000  *u.AA):
    waveList = []
    fluxList = []
    errList  = []
    nameList = []
    
    # Cut wavelength in a more reasonable range
    for s in specList:
        inds = np.where((s.wave.to(u.AA).reshape(1, -1)[0] > wmin) & (s.wave.to(u.AA).reshape(1, -1)[0] < wmax))
        
        nameList.append(str(s.qid) if s.qid is not None else '')
        waveList.append(list(s.wave.to(u.AA).value.reshape(1, -1)[0][inds]))
        fluxList.append(list(s.flux.value.reshape(1, -1)[0][inds]))
        errList.append(list(s.err.value.reshape(1, -1)[0][inds]))

    specDBData = getObservationData(nameList)
    fibreHDU = generateFibresData(specDBData)

    # fix the path for the outfile if I forget a "/" or the extension
    if not outfile.endswith(".fits"):
        outfile = outfile + ".fits"

    if not outpath.endswith("/"):
        outpath = outpath + "/"

    writeFits(fluxList, errList, waveList, fibre = fibreHDU, name = outpath + outfile)


# ---------------------------------------------------------------------- #


# from unit to scalar
def toSc(smth_wunit):
    return smth_wunit.value


# Convert units
def cU(data, u_to, eq=None):
    return data.to(u_to, equivalencies=eq)


class spec:
    @u.quantity_input
    def __init__(self,
                 wave: u.nm,
                 flux: u.W / u.m**2 / u.nm,
                 err: u.W / u.m**2 / u.nm,
                 gid=None,
                 qid=None,
                 ra=None,
                 dec=None,
                 z=None,
                 orig='GaiaCalib',
                 AB=None,
                 eAB=None,
                 filename=None):
        self.wave = wave
        self.flux = flux
        self.err = err
        self.qid = qid
        self.ra = ra
        self.dec = dec
        self.z = z
        self.gid = gid
        self.orig = orig
        self.AB = AB
        self.eAB = eAB
        self.filename = filename

    def write(self, outfile, fmt, overwrite=False):
        if '.' in outfile:
            outfile = outfile.split(".")[0]
        if self.AB is None or self.eAB is None:
            self.toABMag()

        t = Table([
            toSc(self.wave.to(u.AA)),
            toSc(self.flux),
            toSc(self.err), self.AB, self.eAB
        ],
                  names=("wave", "flux", "err", "AB", "eAB"),
                  meta=self.get_meta())
        ext = '.fits' if fmt == 'fits' else '.dat'
        t.write(outfile + ext, format=fmt, overwrite=overwrite)

    def toFrequency(self):
        if self.flux.unit.to_string(
        ) == 'erg / (Angstrom cm2 s)' and self.err.unit.to_string(
        ) == 'erg / (Angstrom cm2 s)':
            return self.flux, self.err
        return cU(self.flux, (u.erg / u.cm**2 / u.AA / u.s)), cU(
            self.err, (u.erg / u.cm**2 / u.AA / u.s))

    def toABMag(self):
        f_nu = self.flux.to(u.Jy,
                            equivalencies=u.spectral_density(self.wave))  # Jy
        ef_nu = self.err.to(u.Jy, equivalencies=u.spectral_density(self.wave))
        self.AB = -2.5 * np.log10(toSc(f_nu) / 3631)
        self.eAB = np.abs(1. / toSc(f_nu) * toSc(ef_nu))
        return -2.5 * np.log10(toSc(f_nu) / 3631), np.abs(1. / toSc(f_nu) *
                                                          toSc(ef_nu))

    def get_meta(self):
        return {
            'qid': self.qid,
            'gid': self.gid,
            'ra': self.ra,
            'dec': self.dec,
            'z': self.z,
            'ORIGIN': self.orig
        }

    def import_meta(self, data):
        idx = np.argwhere(data['source_id'] == self.gid).flatten()
        if len(idx) == 0:
            return
        self.qid = data['qid'][idx][0][0]
        self.ra = data['RAd'][idx][0][0]
        self.dec = data['DECd'][idx][0][0]
        if not 'z_spec' in data.names or np.isnan(data['z_spec'][idx][0]):
            self.z = -1.
        else:
            self.z = data['z_spec'][idx][0][0]

    def plot(self, out_folder=None, out=None):
        _, ax = plt.subplots(1, 1, figsize=(8, 8 / 1.61))
        ax.plot(toSc(self.wave), toSc(self.flux))
        ax.fill_between(toSc(self.wave),
                        toSc(self.flux) + toSc(self.err),
                        toSc(self.flux) - toSc(self.err),
                        alpha=0.25,
                        color='blue')
        ax.plot(toSc(self.wave),
                toSc(self.flux) + toSc(self.err),
                lw=0.5,
                color='grey')
        ax.plot(toSc(self.wave),
                toSc(self.flux) - toSc(self.err),
                lw=0.5,
                color='grey')
        ax.set_xlabel(self.wave.unit.to_string())
        ax.set_ylabel(self.flux.unit.to_string())
        ax.grid(ls='-.', c='lightgrey')
        ax.set_axisbelow(True)
        if out_folder is not None and out is not None:
            plt.savefig(out_folder + '/' + out, bbox_inches='tight')
        else:
            plt.show()

    def incremental_qid(self, path):
        if self.qid is not None:
            return str(self.qid)
        elif self.gid is not None:
            return str(self.gid)
        file_list = os.listdir(path)

        GSpecList = []
        for file in file_list:
            if file.split("_")[0] == 'GaiaSpec':
                GSpecList.append(file)
        GSpecList.sort()

        if not 'GaiaSpec_0' in file_list:
            return 'GaiaSpec_0'
        else:
            return 'GaiaSpec_' + str(int(GSpecList[-1].split('_')[1]) + 1)

    def toMarz(self, outpath, outfile):
        specDBData = getObservationData(self.incremental_qid(outpath))
        fibreHDU = generateFibresData(specDBData)

        writeFits(toSc(self.flux).reshape(1, -1),
                  toSc(self.err).reshape(1, -1),
                  (toSc(self.wave.to(u.AA))).reshape(1, -1),
                  fibre=fibreHDU,
                  name=outpath + '/' + outfile)


def load_spectra(path, wmin=330, wmax=1050, step=1440, space=np.linspace):
    spec_list = []
    for _file in tqdm(os.listdir(path)):
        if _file.endswith(".fits"):
            spec_list.append(
                load_single_spec(path + _file,
                                 wmin=wmin,
                                 wmax=wmax,
                                 step=step,
                                 space=space))
    return spec_list


def load_single_spec(path, wmin=330, wmax=1050, step=1440, space=np.linspace, truncation = False):
    calibrated_df, sampling = gaiaxpy.calibrate(path,
                                                sampling=space(
                                                    wmin, wmax, step),
                                                save_file=False, truncation = truncation)
    return spec(sampling * u.nm,
                calibrated_df['flux'][0] * u.W / u.m**2 / u.nm,
                calibrated_df['flux_error'][0] * u.W / u.m**2 / u.nm,
                gid=calibrated_df['source_id'][0],
                filename=path)


def load_meta_archive(path):
    with fits.open(path) as f:
        meta = f[1].data  # pylint: disable=no-member
    return meta


def import_meta_all(spec_list, meta):
    for _spec in spec_list:
        _spec.import_meta(meta)
    return 0


def plot_all(spec_list, out_folder=None):
    for _spec in spec_list:
        _spec.plot(out_folder=out_folder,
                   out=_spec.incremental_qid(out_folder))
    return 0


def check_unique(spec_list, metadata):
    """Checks if more than one spectrum is associated with a single source."""
    unique = []
    non_unique = []
    non_unique_qid = []
    for _spec in spec_list:
        if _spec.gid is None:
            _spec.import_meta(metadata)

        if _spec.gid in unique:
            non_unique.append(_spec.gid)
            non_unique_qid.append(_spec.qid[0])
        else:
            unique.append(_spec.gid)
    return non_unique_qid, non_unique, unique


def check_missing_meta(spec_list, metadata):
    """Returns a list of spectra with no associated qid"""
    out = []
    for _spec in spec_list:
        if _spec.qid is None:
            _spec.import_meta(metadata)
            if _spec.qid is None:
                out.append(_spec.filename)
    return out


def get_spec_from_archive(ra=None,
                          dec=None,
                          radius=1.5,
                          gid=None,
                          qid=None,
                          wmin=330,
                          wmax=1050,
                          step=1440,
                          space=np.linspace):
    if gid is not None:
        query = "SELECT gaia_source.source_id FROM gaiadr3.gaia_source WHERE source_id = {}".format(
            gid)
    elif gid is None and ra is not None and dec is not None:
        query = """SELECT gaia_source.source_id FROM gaiadr3.gaia_source WHERE CONTAINS(
        POINT('ICRS', gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec), CIRCLE('ICRS', {}, {}, {})) = 1""".format(
            ra, dec, radius / 3600)
    else:
        print("Please enter either RA/DEC or source_id!")

    calibrated_df, sampling = gaiaxpy.calibrate(query,
                                                sampling=space(
                                                    wmin, wmax, step),
                                                save_file=False)
    out_spec = spec(sampling * u.nm,
                    calibrated_df['flux'][0] * u.W / u.m**2 / u.nm,
                    calibrated_df['flux_error'][0] * u.W / u.m**2 / u.nm,
                    gid=calibrated_df['source_id'][0],
                    qid=qid)
    return out_spec


def get_spec_dict(spec_list):
    out_qid, out_gid = {}, {}
    for _spec in spec_list:
        out_qid[_spec.qid] = _spec
        out_gid[_spec.gid] = _spec
    return out_qid, out_gid


def get_spec_by_sourceid(spec_dict, meta):
    for _qid, _gid in zip(meta['qid'], meta['source_id']):
        pass