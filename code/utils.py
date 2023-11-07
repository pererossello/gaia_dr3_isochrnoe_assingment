import numpy as np

def get_hist_scatter_colors(x_rad, y_rad, 
                            x_lims=[0, 2*np.pi], 
                            y_lims=[-np.pi/2, np.pi/2], 
                            bins=[180, 90]):
    fact = 3
    h, xedges, yedges = np.histogram2d(x_rad, y_rad, 
                                       bins=[np.linspace(x_lims[0], x_lims[1], bins[0]*fact), 
                                             np.linspace(y_lims[0], y_lims[1], bins[1]*fact)])
    xcenters = (xedges[:-1] + xedges[1:]) / 2 - np.pi
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # Determine the bin indices for each point
    x_indices = np.digitize(x_rad, xedges) - 1
    y_indices = np.digitize(y_rad, yedges) - 1

    # Ensure indices are within the valid range
    x_indices = np.clip(x_indices, 0, len(xcenters) - 1)
    y_indices = np.clip(y_indices, 0, len(ycenters) - 1)

    # Retrieve the density values for each point
    colors = h.T[y_indices, x_indices]

    return colors


def preprocess_data(data):

    data['distance'] = 1e3 / data['parallax']

    data['g_abs'] = data['phot_g_mean_mag'] + 5 * np.log10(data['parallax'] / 100.0)

    data['phot_g_mean_mag_error'] = sigma_g_mag = (2.5 / np.log(10)) * (data['phot_g_mean_flux_error'] / data['phot_g_mean_flux'])
    data['g_abs_error'] = np.sqrt(data['phot_g_mean_mag_error']**2 + 
                                (5 / (data['parallax'] * np.log(10)) * data['parallax_error'])**2)

    bp_magnitude = -2.5 * np.log10(data['phot_bp_mean_flux'])
    rp_magnitude = -2.5 * np.log10(data['phot_rp_mean_flux'])
    color_magnitude = bp_magnitude - rp_magnitude 
    cal = data['bp_rp'] - color_magnitude
    color_magnitude = color_magnitude + cal
    data['bp_rp_mag'] = color_magnitude

    sigma_bp_mag = (2.5 / np.log(10)) * (data['phot_bp_mean_flux_error'] / data['phot_bp_mean_flux'])
    sigma_rp_mag = (2.5 / np.log(10)) * (data['phot_rp_mean_flux_error'] / data['phot_rp_mean_flux'])
    sigma_color = np.sqrt(sigma_bp_mag**2 + sigma_rp_mag**2)
    data['bp_rp_mag_error'] = sigma_color

    return data



# Coordinates of clusters. Angles are in degrees and distances in parsecs
cluster_dic = {
    'Hyades': {
        'distance': 47.0,
        'err_distance': 0,
        'dec_center': 15.87,
        'ra_center': 66.75,
        'tidal_radius': 10,
        'core_radius': 2.7
    },
    '47 Tucanae': { 
        'distance': 4e3,
        'err_distance': 0.35e3,
        'dec_center': -72 - 4/60 - 52.6/3600,
        'ra_center': 15 * (0 + 24/60 + 5.67/3600) ,
        'tidal_radius': 18.4,
        'core_radius': 18.4
    }
}


def cluster_query(cluster):
 
    var = 'core_radius'
    ang_radius = np.arctan(cluster[var]/cluster['distance']) * 180/np.pi

    parallax_min = 1e3 / (cluster['distance'] + cluster[var] + cluster['err_distance'])
    parallax_max = 1e3 / (cluster['distance'] - cluster[var] - cluster['err_distance'])

    adql_query = f"""
    SELECT dr3.ra, dr3.dec, dr3.ra_error, dr3.dec_error,
    dr3.parallax, dr3.parallax_error, dr3.parallax_over_error,
    dr3.phot_g_mean_flux, dr3.phot_g_mean_mag, dr3.phot_g_mean_flux_error, 
    dr3.bp_rp, 
    dr3.phot_rp_mean_flux, dr3.phot_rp_mean_mag, dr3.phot_rp_mean_flux_error,
    dr3.phot_bp_mean_flux, dr3.phot_bp_mean_flux_error,
    dr3.astrometric_excess_noise, dr3.astrometric_excess_noise_sig,
    dr3.phot_bp_rp_excess_factor,
    dr3.pm, dr3.pmra, dr3.pmra_error, dr3.pmdec, dr3.pmdec_error, 
    dr3.phot_bp_mean_flux_over_error, dr3.phot_rp_mean_flux_over_error,
    dr3.l, dr3.b,
    dr3.ag_gspphot, dr3.ag_gspphot_lower, dr3.ag_gspphot_upper,
    dr3.ebpminrp_gspphot, dr3.ebpminrp_gspphot_lower, dr3.ebpminrp_gspphot_upper
    FROM gaiadr3.gaia_source AS dr3
    WHERE 1=CONTAINS(
        POINT('ICRS', dr3.ra, dr3.dec),
        CIRCLE('ICRS', {cluster['ra_center']}, {cluster['dec_center']}, {ang_radius}))
    AND dr3.parallax BETWEEN {parallax_min} AND {parallax_max}
    AND dr3.parallax_over_error > 0
    AND dr3.phot_bp_mean_flux_over_error > 0
    AND dr3.phot_rp_mean_flux_over_error > 0
    """

    return adql_query



def distance_query(distance):
 
    par = 1e3/distance

    adql_query = f"""
    SELECT dr3.ra, dr3.dec, dr3.ra_error, dr3.dec_error,
    dr3.parallax, dr3.parallax_error, dr3.parallax_over_error,
    dr3.phot_g_mean_flux, dr3.phot_g_mean_mag, dr3.phot_g_mean_flux_error, 
    dr3.bp_rp, 
    dr3.phot_rp_mean_flux, dr3.phot_rp_mean_mag, dr3.phot_rp_mean_flux_error,
    dr3.phot_bp_mean_flux, dr3.phot_bp_mean_flux_error,
    dr3.astrometric_excess_noise, dr3.astrometric_excess_noise_sig,
    dr3.phot_bp_rp_excess_factor,
    dr3.pm, dr3.pmra, dr3.pmra_error, dr3.pmdec, dr3.pmdec_error, 
    dr3.phot_bp_mean_flux_over_error, dr3.phot_rp_mean_flux_over_error,
    dr3.l, dr3.b,
    dr3.non_single_star,
    dr3.ag_gspphot, dr3.ag_gspphot_lower, dr3.ag_gspphot_upper,
    dr3.ebpminrp_gspphot, dr3.ebpminrp_gspphot_lower, dr3.ebpminrp_gspphot_upper
    FROM gaiadr3.gaia_source AS dr3
    WHERE dr3.parallax > {par}
    AND dr3.parallax_over_error > 0
    AND dr3.phot_bp_mean_flux_over_error > 0
    AND dr3.phot_rp_mean_flux_over_error > 0
    """

    return adql_query


def distance_query_(distance):
 
    par = 1e3/distance

    adql_query = f"""
    SELECT dr3.ra, dr3.dec, dr3.ra_error, dr3.dec_error,
    dr3.parallax, dr3.parallax_error, dr3.parallax_over_error,
    dr3.phot_g_mean_flux, dr3.phot_g_mean_mag, dr3.phot_g_mean_flux_error, 
    dr3.bp_rp, 
    dr3.phot_rp_mean_flux, dr3.phot_rp_mean_mag, dr3.phot_rp_mean_flux_error,
    dr3.phot_bp_mean_flux, dr3.phot_bp_mean_flux_error,
    dr3.astrometric_excess_noise, dr3.astrometric_excess_noise_sig,
    dr3.phot_bp_rp_excess_factor,
    dr3.pm, dr3.pmra, dr3.pmra_error, dr3.pmdec, dr3.pmdec_error, 
    dr3.phot_bp_mean_flux_over_error, dr3.phot_rp_mean_flux_over_error,
    dr3.l, dr3.b,
    dr3.non_single_star,
    dr3.ag_gspphot, dr3.ag_gspphot_lower, dr3.ag_gspphot_upper,
    dr3.ebpminrp_gspphot, dr3.ebpminrp_gspphot_lower, dr3.ebpminrp_gspphot_upper
    FROM gaiadr3.gaia_source AS dr3
    WHERE dr3.parallax > {par}
    AND dr3.parallax_over_error > 10
    AND dr3.phot_bp_mean_flux_over_error > 10
    AND dr3.phot_rp_mean_flux_over_error > 10
    AND dr3.astrometric_excess_noise < 1
    """

    return adql_query