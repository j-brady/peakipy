from enum import Enum

import pandas as pd
from numpy import sqrt, exp, log
from scipy.special import wofz

from peakipy.constants import π, tiny, log2


class Lineshape(str, Enum):
    PV = "PV"
    V = "V"
    G = "G"
    L = "L"
    PV_PV = "PV_PV"
    G_L = "G_L"
    PV_G = "PV_G"
    PV_L = "PV_L"


def gaussian(x, center=0.0, sigma=1.0):
    r"""1-dimensional Gaussian function.

    gaussian(x, center, sigma) =
        (1/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    :math:`\\frac{1}{ \sqrt{2\pi} } exp \left( \\frac{-(x-center)^2}{2 \sigma^2} \\right)`

    :param x: x
    :param center: center
    :param sigma: sigma
    :type x: numpy.array
    :type center: float
    :type sigma: float

    :return: 1-dimensional Gaussian
    :rtype: numpy.array

    """
    return (1.0 / max(tiny, (sqrt(2 * π) * sigma))) * exp(
        -((1.0 * x - center) ** 2) / max(tiny, (2 * sigma**2))
    )


def lorentzian(x, center=0.0, sigma=1.0):
    r"""1-dimensional Lorentzian function.

    lorentzian(x, center, sigma) =
        (1/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)

    :math:`\\frac{1}{ 1+ \left( \\frac{x-center}{\sigma}\\right)^2} / (\pi\sigma)`

    :param x: x
    :param center: center
    :param sigma: sigma
    :type x: numpy.array
    :type center: float
    :type sigma: float

    :return: 1-dimensional Lorenztian
    :rtype: numpy.array

    """
    return (1.0 / (1 + ((1.0 * x - center) / max(tiny, sigma)) ** 2)) / max(
        tiny, (π * sigma)
    )


def voigt(x, center=0.0, sigma=1.0, gamma=None):
    r"""Return a 1-dimensional Voigt function.

    voigt(x, center, sigma, gamma) =
        amplitude*wofz(z).real / (sigma*sqrt(2.0 * π))

    :math:`V(x,\sigma,\gamma) = (\\frac{Re[\omega(z)]}{\sigma \sqrt{2\pi}})`

    :math:`z=\\frac{x+i\gamma}{\sigma\sqrt{2}}`

    see Voigt_ wiki

    .. _Voigt: https://en.wikipedia.org/wiki/Voigt_profile


    :param x: x values
    :type x: numpy array 1d
    :param center: center of lineshape in points
    :type center: float
    :param sigma: sigma of gaussian
    :type sigma: float
    :param gamma: gamma of lorentzian
    :type gamma: float

    :returns: Voigt lineshape
    :rtype: numpy.array

    """
    if gamma is None:
        gamma = sigma

    z = (x - center + 1j * gamma) / max(tiny, (sigma * sqrt(2.0)))
    return wofz(z).real / max(tiny, (sigma * sqrt(2.0 * π)))


def pseudo_voigt(x, center=0.0, sigma=1.0, fraction=0.5):
    r"""1-dimensional Pseudo-voigt function

    Superposition of Gaussian and Lorentzian function

    :math:`(1-\phi) G(x,center,\sigma_g) + \phi L(x, center, \sigma)`

    Where :math:`\phi` is the fraction of Lorentzian lineshape and :math:`G` and :math:`L` are Gaussian and
    Lorentzian functions, respectively.

    :param x: data
    :type x: numpy.array
    :param center: center of peak
    :type center: float
    :param sigma: sigma of lineshape
    :type sigma: float
    :param fraction: fraction of lorentzian lineshape (between 0 and 1)
    :type fraction: float

    :return: pseudo-voigt function
    :rtype: numpy.array

    """
    sigma_g = sigma / sqrt(2 * log2)
    pv = (1 - fraction) * gaussian(x, center, sigma_g) + fraction * lorentzian(
        x, center, sigma
    )
    return pv


def pvoigt2d(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    r"""2D pseudo-voigt model

    :math:`(1-fraction) G(x,center,\sigma_{gx}) + (fraction) L(x, center, \sigma_x) * (1-fraction) G(y,center,\sigma_{gy}) + (fraction) L(y, center, \sigma_y)`

    :param XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
    :type XY: numpy.array

    :param amplitude: amplitude of peak
    :type amplitude: float

    :param center_x: center of peak in x
    :type center_x: float

    :param center_y: center of peak in x
    :type center_y: float

    :param sigma_x: sigma of lineshape in x
    :type sigma_x: float

    :param sigma_y: sigma of lineshape in y
    :type sigma_y: float

    :param fraction: fraction of lorentzian lineshape (between 0 and 1)
    :type fraction: float

    :return: flattened array of Z values (use Z.reshape(X.shape) for recovery)
    :rtype: numpy.array

    """
    x, y = XY
    pv_x = pseudo_voigt(x, center_x, sigma_x, fraction)
    pv_y = pseudo_voigt(y, center_y, sigma_y, fraction)
    return amplitude * pv_x * pv_y


def pv_l(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    """2D lineshape model with pseudo-voigt in x and lorentzian in y

    Arguments
    =========

        -- XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
        -- amplitude: peak amplitude (gaussian and lorentzian)
        -- center_x: position of peak in x
        -- center_y: position of peak in y
        -- sigma_x: linewidth in x
        -- sigma_y: linewidth in y
        -- fraction: fraction of lorentzian in fit

    Returns
    =======

        -- flattened array of Z values (use Z.reshape(X.shape) for recovery)

    """

    x, y = XY
    pv_x = pseudo_voigt(x, center_x, sigma_x, fraction)
    pv_y = pseudo_voigt(y, center_y, sigma_y, 1.0)  # lorentzian
    return amplitude * pv_x * pv_y


def pv_g(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    """2D lineshape model with pseudo-voigt in x and gaussian in y

    Arguments
    ---------

        -- XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
        -- amplitude: peak amplitude (gaussian and lorentzian)
        -- center_x: position of peak in x
        -- center_y: position of peak in y
        -- sigma_x: linewidth in x
        -- sigma_y: linewidth in y
        -- fraction: fraction of lorentzian in fit

    Returns
    -------

        -- flattened array of Z values (use Z.reshape(X.shape) for recovery)

    """
    x, y = XY
    pv_x = pseudo_voigt(x, center_x, sigma_x, fraction)
    pv_y = pseudo_voigt(y, center_y, sigma_y, 0.0)  # gaussian
    return amplitude * pv_x * pv_y


def pv_pv(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction_x=0.5,
    fraction_y=0.5,
):
    """2D lineshape model with pseudo-voigt in x and pseudo-voigt in y
    i.e. fraction_x and fraction_y params

    Arguments
    =========

        -- XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
        -- amplitude: peak amplitude (gaussian and lorentzian)
        -- center_x: position of peak in x
        -- center_y: position of peak in y
        -- sigma_x: linewidth in x
        -- sigma_y: linewidth in y
        -- fraction_x: fraction of lorentzian in x
        -- fraction_y: fraction of lorentzian in y

    Returns
    =======

        -- flattened array of Z values (use Z.reshape(X.shape) for recovery)

    """

    x, y = XY
    pv_x = pseudo_voigt(x, center_x, sigma_x, fraction_x)
    pv_y = pseudo_voigt(y, center_y, sigma_y, fraction_y)
    return amplitude * pv_x * pv_y


def gaussian_lorentzian(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    fraction=0.5,
):
    """2D lineshape model with gaussian in x and lorentzian in y

    Arguments
    =========

        -- XY: meshgrid of X and Y coordinates [X,Y] each with shape Z
        -- amplitude: peak amplitude (gaussian and lorentzian)
        -- center_x: position of peak in x
        -- center_y: position of peak in y
        -- sigma_x: linewidth in x
        -- sigma_y: linewidth in y
        -- fraction: fraction of lorentzian in fit

    Returns
    =======

        -- flattened array of Z values (use Z.reshape(X.shape) for recovery)

    """
    x, y = XY
    pv_x = pseudo_voigt(x, center_x, sigma_x, 0.0)  # gaussian
    pv_y = pseudo_voigt(y, center_y, sigma_y, 1.0)  # lorentzian
    return amplitude * pv_x * pv_y


def voigt2d(
    XY,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    sigma_x=1.0,
    sigma_y=1.0,
    gamma_x=1.0,
    gamma_y=1.0,
    fraction=0.5,
):
    fraction = 0.5
    gamma_x = None
    gamma_y = None
    x, y = XY
    voigt_x = voigt(x, center_x, sigma_x, gamma_x)
    voigt_y = voigt(y, center_y, sigma_y, gamma_y)
    return amplitude * voigt_x * voigt_y


def get_lineshape_function(lineshape: Lineshape):
    match lineshape:
        case lineshape.PV | lineshape.G | lineshape.L:
            lineshape_function = pvoigt2d
        case lineshape.V:
            lineshape_function = voigt2d
        case lineshape.PV_PV:
            lineshape_function = pv_pv
        case lineshape.G_L:
            lineshape_function = gaussian_lorentzian
        case lineshape.PV_G:
            lineshape_function = pv_g
        case lineshape.PV_L:
            lineshape_function = pv_l
        case _:
            raise Exception("No lineshape was selected!")
    return lineshape_function


def calculate_height_for_voigt_lineshape(df):
    df["height"] = df.apply(
        lambda x: voigt2d(
            XY=[0, 0],
            center_x=0.0,
            center_y=0.0,
            sigma_x=x.sigma_x,
            sigma_y=x.sigma_y,
            gamma_x=x.gamma_x,
            gamma_y=x.gamma_y,
            amplitude=x.amp,
        ),
        axis=1,
    )
    df["height_err"] = df.apply(
        lambda x: x.amp_err * (x.height / x.amp) if x.amp_err != None else 0.0,
        axis=1,
    )
    return df


def calculate_fwhm_for_voigt_lineshape(df):
    df["fwhm_g_x"] = df.sigma_x.apply(
        lambda x: 2.0 * x * sqrt(2.0 * log(2.0))
    )  # fwhm of gaussian
    df["fwhm_g_y"] = df.sigma_y.apply(lambda x: 2.0 * x * sqrt(2.0 * log(2.0)))
    df["fwhm_l_x"] = df.gamma_x.apply(lambda x: 2.0 * x)  # fwhm of lorentzian
    df["fwhm_l_y"] = df.gamma_y.apply(lambda x: 2.0 * x)
    df["fwhm_x"] = df.apply(
        lambda x: 0.5346 * x.fwhm_l_x
        + sqrt(0.2166 * x.fwhm_l_x**2.0 + x.fwhm_g_x**2.0),
        axis=1,
    )
    df["fwhm_y"] = df.apply(
        lambda x: 0.5346 * x.fwhm_l_y
        + sqrt(0.2166 * x.fwhm_l_y**2.0 + x.fwhm_g_y**2.0),
        axis=1,
    )
    return df


def calculate_height_for_pseudo_voigt_lineshape(df):
    df["height"] = df.apply(
        lambda x: pvoigt2d(
            XY=[0, 0],
            center_x=0.0,
            center_y=0.0,
            sigma_x=x.sigma_x,
            sigma_y=x.sigma_y,
            amplitude=x.amp,
            fraction=x.fraction,
        ),
        axis=1,
    )
    df["height_err"] = df.apply(lambda x: x.amp_err * (x.height / x.amp), axis=1)
    return df


def calculate_fwhm_for_pseudo_voigt_lineshape(df):
    df["fwhm_x"] = df.sigma_x.apply(lambda x: x * 2.0)
    df["fwhm_y"] = df.sigma_y.apply(lambda x: x * 2.0)
    return df


def calculate_height_for_gaussian_lineshape(df):
    df["height"] = df.apply(
        lambda x: pvoigt2d(
            XY=[0, 0],
            center_x=0.0,
            center_y=0.0,
            sigma_x=x.sigma_x,
            sigma_y=x.sigma_y,
            amplitude=x.amp,
            fraction=0.0,  # gaussian
        ),
        axis=1,
    )
    df["height_err"] = df.apply(lambda x: x.amp_err * (x.height / x.amp), axis=1)
    return df


def calculate_height_for_lorentzian_lineshape(df):
    df["height"] = df.apply(
        lambda x: pvoigt2d(
            XY=[0, 0],
            center_x=0.0,
            center_y=0.0,
            sigma_x=x.sigma_x,
            sigma_y=x.sigma_y,
            amplitude=x.amp,
            fraction=1.0,  # lorentzian
        ),
        axis=1,
    )
    df["height_err"] = df.apply(lambda x: x.amp_err * (x.height / x.amp), axis=1)
    return df


def calculate_height_for_pv_pv_lineshape(df):
    df["height"] = df.apply(
        lambda x: pv_pv(
            XY=[0, 0],
            center_x=0.0,
            center_y=0.0,
            sigma_x=x.sigma_x,
            sigma_y=x.sigma_y,
            amplitude=x.amp,
            fraction_x=x.fraction_x,
            fraction_y=x.fraction_y,
        ),
        axis=1,
    )
    df["height_err"] = df.apply(lambda x: x.amp_err * (x.height / x.amp), axis=1)
    return df


def calculate_peak_centers_in_ppm(df, peakipy_data):
    #  convert values to ppm
    df["center_x_ppm"] = df.center_x.apply(lambda x: peakipy_data.uc_f2.ppm(x))
    df["center_y_ppm"] = df.center_y.apply(lambda x: peakipy_data.uc_f1.ppm(x))
    df["init_center_x_ppm"] = df.init_center_x.apply(
        lambda x: peakipy_data.uc_f2.ppm(x)
    )
    df["init_center_y_ppm"] = df.init_center_y.apply(
        lambda x: peakipy_data.uc_f1.ppm(x)
    )
    return df


def calculate_peak_linewidths_in_hz(df, peakipy_data):
    df["sigma_x_ppm"] = df.sigma_x.apply(lambda x: x * peakipy_data.ppm_per_pt_f2)
    df["sigma_y_ppm"] = df.sigma_y.apply(lambda x: x * peakipy_data.ppm_per_pt_f1)
    df["fwhm_x_ppm"] = df.fwhm_x.apply(lambda x: x * peakipy_data.ppm_per_pt_f2)
    df["fwhm_y_ppm"] = df.fwhm_y.apply(lambda x: x * peakipy_data.ppm_per_pt_f1)
    df["fwhm_x_hz"] = df.fwhm_x.apply(lambda x: x * peakipy_data.hz_per_pt_f2)
    df["fwhm_y_hz"] = df.fwhm_y.apply(lambda x: x * peakipy_data.hz_per_pt_f1)
    return df


def calculate_lineshape_specific_height_and_fwhm(
    lineshape: Lineshape, df: pd.DataFrame
):
    match lineshape:
        case lineshape.V:
            df = calculate_height_for_voigt_lineshape(df)
            df = calculate_fwhm_for_voigt_lineshape(df)

        case lineshape.PV:
            df = calculate_height_for_pseudo_voigt_lineshape(df)
            df = calculate_fwhm_for_pseudo_voigt_lineshape(df)

        case lineshape.G:
            df = calculate_height_for_gaussian_lineshape(df)
            df = calculate_fwhm_for_pseudo_voigt_lineshape(df)

        case lineshape.L:
            df = calculate_height_for_lorentzian_lineshape(df)
            df = calculate_fwhm_for_pseudo_voigt_lineshape(df)

        case lineshape.PV_PV:
            df = calculate_height_for_pv_pv_lineshape(df)
            df = calculate_fwhm_for_pseudo_voigt_lineshape(df)
        case _:
            df = calculate_fwhm_for_pseudo_voigt_lineshape(df)
    return df
