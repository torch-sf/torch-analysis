import yt
import numpy as np

# Initialize quantity dictionaries
QUANTITY_REGISTRY = {}
QUANTITY_TYPE = {}
QUANTITY_LABELS = {}
QUANTITY_REQUISITES = {}

# =============================================================================
# Helper functions
# =============================================================================

def _get_roi_region(ds, parfile="flash.par"):
    """
    Return yt region corresponding to derefinement ROI as defined in flash.par.
    For roi gas quantities, pass this container instead of ds which takes .all_data().
    Internal function, not registered for users. 
    """
    from torch_param import FlashPar  
    flashp = FlashPar(parfile)

    left_edge = ds.arr(
        [
            flashp["deref_xl"],
            flashp["deref_yl"],
            flashp["deref_zl"],
        ],
        "cm",
    )

    right_edge = ds.arr(
        [
            flashp["deref_xr"],
            flashp["deref_yr"],
            flashp["deref_zr"],
        ],
        "cm",
    )

    center = 0.5 * (left_edge + right_edge)

    return ds.region(
        center=center,
        left_edge=left_edge,
        right_edge=right_edge,
    )


def _gas_mass_container(container):
    """Compute total mass of gas in the container in Msun.
        Internal function, not registered for users. 
    """
    return (
        container[("gas", "mass")]
        .sum()
        .to("Msun")
        .value
    )


def _gas_virial_ratio_container(container):
    """Computes virial ratio of gas in the container defined as |sum(K)/sum(U)|.
    The potential energy U uses only the gas self-gravity potential (gpot); the
    stellar potential (bgpt) is intentionally excluded.
        Internal function, not registered for users.
    """
    dens = container[('flash','dens')]
    vx = container[('flash','velx')]
    vy = container[('flash','vely')]
    vz = container[('flash','velz')]
    v2 = vx**2 + vy**2 + vz**2

    # Weight the per-cell energy densities by cell volume before summing, since
    # cells vary in size on the AMR grid.
    vol = container[('gas','cell_volume')]

    Ekin = 0.5*dens*v2*vol
    # Potential energy from gas self-gravity only (gpot); the stellar
    # potential (bgpt) is intentionally excluded for the gas virial ratio.
    gas_gas_gpot = container[('flash','gpot')]
    Epot = dens*gas_gas_gpot.v*vol

    alpha = abs(sum(Ekin.v)/sum(Epot.v))

    return alpha


def _bound_gas_mass_fraction_container(container):
    """Computes fraction of bound gas over total gas mass in the yt container.
    Internal function, not registered for users. 
    """
    mass = container[('gas','mass')]
    dens = container[('flash','dens')]
    vx = container[('flash','velx')]
    vy = container[('flash','vely')]
    vz = container[('flash','velz')]
    v2 = vx**2 + vy**2 + vz**2

    Bx = container[('flash','magx')]
    By = container[('flash','magy')]
    Bz = container[('flash','magz')]
    B2 = Bx**2 + By**2 + Bz**2

    # Magnetic energy
    Emag = 0.5*B2
    # Kinetic energy
    Ekin = 0.5*dens*v2
    # Potential energy from stars and gas+sinks
    gas_gas_gpot = container[('flash','gpot')]
    star_gas_gpot = container[('flash','bgpt')]
    Epot = dens*(gas_gas_gpot.v+star_gas_gpot.v)
    # Thermal energy TODO: generalize gamma
    Ethe = 3.0/2.0*container[("flash","pres")]
    # Total energy
    Etot = (Ekin.v+Epot.v+Emag.v+Ethe.v)

    bound_idx = np.where(Etot < 0.0)
    
    bmf = sum(mass[bound_idx])/sum(mass)

    return bmf


def _particle_mass_container(ds, particle_type="all"):
    """
    Return particle masses (Msun) in the dataset for the given particle type.
    Internal function, not registered for users.

    Parameters
    ----------
    ds : yt Dataset
        The dataset to analyze.
    particle_type : str
        - 'all'   : the summed mass of all particles (a scalar)
        - 'stars' : per-particle masses of stars (particle_csgm == 0)
        - 'sinks' : per-particle masses of sinks (particle_csgm != 0)

    Returns
    -------
    float or numpy.ndarray
        For 'all', the total mass in Msun (0.0 if there are none). For
        'stars'/'sinks', an array of individual masses in Msun (an empty
        array if there are no particles of that type). Callers apply
        .sum()/len()/max() as needed.
    """
    if not ds.particles_exist:
        return np.array([])

    ad = ds.all_data()

    pm = ad[("all", "particle_mass")].to("Msun").value

    if particle_type == "all":
        return pm.sum() if pm.size > 0 else 0.0

    # Get particle type mask
    csgm = ad[("all", "particle_csgm")].value
    if particle_type == "stars":
        mask = (csgm == 0)
    elif particle_type == "sinks":
        mask = (csgm != 0)
    else:
        raise ValueError(f"Unknown particle_type '{particle_type}'")

    # Return an (empty) array so callers can uniformly use len()/.sum()/max(...)
    return pm[mask]


def _ellipticity_from_inertial_tensor(mass, px, py, pz):
    """
    Compute ellipticity c/a for a set of point masses using the reduced
    inertia (shape) tensor method. `mass` is in Msun and `px, py, pz` are
    positions in pc (units are irrelevant to the ratio, but keep them
    consistent). Returns c/a where a >= b >= c are the principal axes, or
    0.0 if there are no points.
    See: https://doi.org/10.1093/mnras/sty3531
    Internal function, not registered for users.
    """
    mass = np.asarray(mass)
    if len(mass) == 0:
        return 0.0

    # Center of mass
    total_mass = np.sum(mass)
    com_x = np.sum(mass * px) / total_mass
    com_y = np.sum(mass * py) / total_mass
    com_z = np.sum(mass * pz) / total_mass

    # Relative positions
    dx = px - com_x
    dy = py - com_y
    dz = pz - com_z

    # Squared distance for the reduced-tensor weighting. Floor away from zero
    # to avoid division by zero for points exactly at the center of mass.
    r2 = dx**2 + dy**2 + dz**2
    r2 = np.maximum(r2, 1e-10)

    # Reduced shape tensor S_ij = sum(m * r_i * r_j / r^2) / sum(m). The sum(m)
    # denominator is optional for the eigenvalue ratio but keeps values normalized.
    weight = mass / r2
    S_xx = np.sum(weight * dx * dx) / total_mass
    S_yy = np.sum(weight * dy * dy) / total_mass
    S_zz = np.sum(weight * dz * dz) / total_mass
    S_xy = np.sum(weight * dx * dy) / total_mass
    S_xz = np.sum(weight * dx * dz) / total_mass
    S_yz = np.sum(weight * dy * dz) / total_mass

    I = np.array([[S_xx, S_xy, S_xz],
                  [S_xy, S_yy, S_yz],
                  [S_xz, S_yz, S_zz]])

    # Eigenvalues are proportional to a^2, b^2, c^2
    eigenvalues = np.sort(np.linalg.eigvalsh(I))[::-1]  # descending: a >= b >= c

    if eigenvalues[0] > 0:
        return np.sqrt(eigenvalues[2] / eigenvalues[0])
    return 0.0


# =============================================================================
# Global quantities
# =============================================================================

def gas_mass(ds):
    """Computes total gas mass on the grid in Msun."""
    return _gas_mass_container(ds.all_data())

QUANTITY_REGISTRY["gas_mass"] = gas_mass
QUANTITY_TYPE["gas_mass"] = 'scalar'
QUANTITY_LABELS["gas_mass"] = r"$M_\mathrm{gas}\,[M_\odot]$"


def gas_virial_ratio(ds):
    """Computes virial ratio of all gas on the grid, defined as |sum(K)/sum(U)|.
    Uses only the gas self-gravity potential (gpot); the stellar potential (bgpt)
    is excluded."""
    return _gas_virial_ratio_container(ds.all_data())

QUANTITY_REGISTRY["gas_virial_ratio"] = gas_virial_ratio
QUANTITY_TYPE["gas_virial_ratio"] = 'scalar'
QUANTITY_LABELS["gas_virial_ratio"] = r"$\alpha_{\rm gas}$"


def bound_gas_mass_fraction(ds):
    """Computes fraction of bound gas over total gas mass in the entire grid."""
    return _bound_gas_mass_fraction_container(ds.all_data())

QUANTITY_REGISTRY["bound_gas_mass_fraction"] = bound_gas_mass_fraction
QUANTITY_TYPE["bound_gas_mass_fraction"] = 'scalar'
QUANTITY_LABELS["bound_gas_mass_fraction"] = r"$M_\mathrm{gas, bound}/M_{\rm tot}$"


def gas_ellipticity(ds):
    """
    Compute gas ellipticity using the reduced inertia tensor method.
    See: https://doi.org/10.1093/mnras/sty3531
    Returns c/a where a, b, c are the principal axes (a >= b >= c).
    Uses lower density cutoff of 1e-20 g/cc.
    """
    ad = ds.all_data()

    # Get gas properties above a density cutoff
    rho = ad[('gas', 'density')].value
    density_cut = 1e-20
    mask = rho > density_cut

    mass = ad[('gas', 'cell_mass')][mask].to('Msun').value
    px = ad[('gas', 'x')][mask].to('pc').value
    py = ad[('gas', 'y')][mask].to('pc').value
    pz = ad[('gas', 'z')][mask].to('pc').value

    return _ellipticity_from_inertial_tensor(mass, px, py, pz)

QUANTITY_REGISTRY["gas_ellipticity"] = gas_ellipticity
QUANTITY_TYPE["gas_ellipticity"] = 'scalar'
QUANTITY_LABELS["gas_ellipticity"] = r"$e_{\rm gas}$"


def sink_mass(ds):
    """Computes total mass of sinks on grid in Msun."""
    if not ds.particles_exist:
        return 0.0
    else:
        return _particle_mass_container(ds, particle_type="sinks").sum()

QUANTITY_REGISTRY["sink_mass"] = sink_mass
QUANTITY_TYPE["sink_mass"] = 'scalar'
QUANTITY_LABELS["sink_mass"] = r"$M_\mathrm{sink}\,[M_\odot]$"


def stellar_mass(ds):
    """Computes total mass of stars on grid in Msun."""
    if not ds.particles_exist:
        return 0.0
    else:
        return _particle_mass_container(ds, particle_type="stars").sum()

QUANTITY_REGISTRY["stellar_mass"] = stellar_mass
QUANTITY_TYPE["stellar_mass"] = 'scalar'
QUANTITY_LABELS["stellar_mass"] = r"$M_\star\,[M_\odot]$"


def stellar_velocity_dispersion(ds):
    """Computes velocity dispersion of stars in cm/s."""
    if not ds.particles_exist:
        return 0.0

    ad = ds.all_data()

    stars = ad[("all", "particle_csgm")].value == 0.0 # star marker

    vx = ad['all','particle_velx'][stars].v
    vy = ad['all','particle_vely'][stars].v
    vz = ad['all','particle_velz'][stars].v
    v2 = vx**2 + vy**2 + vz**2

    return np.std(np.sqrt(v2))

QUANTITY_REGISTRY["stellar_velocity_dispersion"] = stellar_velocity_dispersion
QUANTITY_TYPE["stellar_velocity_dispersion"] = 'scalar'
QUANTITY_LABELS["stellar_velocity_dispersion"] = r"$\sigma_v~[{\rm cm/s}]$"


def stellar_virial_ratio(ds):
    """Computes virial ratio of stars defined as |sum(K)/sum(U)|. Uses only the
    stellar potential (bgpt); the gas potential is excluded."""
    if not ds.particles_exist:
        return 0.0

    ad = ds.all_data()

    stars = ad[("all", "particle_csgm")].value == 0.0 # star marker

    pos = np.array([ad['all','particle_posx'][stars].v,
                               ad['all','particle_posy'][stars].v,
                               ad['all','particle_posz'][stars].v]).T

    m  = ad['all','particle_mass'][stars].v
    vx = ad['all','particle_velx'][stars].v
    vy = ad['all','particle_vely'][stars].v
    vz = ad['all','particle_velz'][stars].v
    v2 = vx**2 + vy**2 + vz**2

    Ekin = 0.5*m*v2
    Epot = m*ds.find_field_values_at_points('bgpt', pos*yt.units.cm).v

    return abs(sum(Ekin)/sum(Epot))

QUANTITY_REGISTRY["stellar_virial_ratio"] = stellar_virial_ratio
QUANTITY_TYPE["stellar_virial_ratio"] = 'scalar'
QUANTITY_LABELS["stellar_virial_ratio"] = r"$\alpha_{\star}$"


def stellar_density(ds):
    """
    Compute half-mass stellar density of the star cluster in Msun/pc^3.
    """
    if not ds.particles_exist:
        return 0.0

    ad = ds.all_data()

    stars = ad[("all", "particle_csgm")].value == 0.0 # star marker

    # Star positions in pc and masses in Msun
    px = ad[("all", "particle_position_x")][stars].to("pc").value
    py = ad[("all", "particle_position_y")][stars].to("pc").value
    pz = ad[("all", "particle_position_z")][stars].to("pc").value
    pm = ad[("all", "particle_mass")][stars].to("Msun").value

    tot_mass = np.sum(pm)
    comx = np.sum(pm*px)/tot_mass
    comy = np.sum(pm*py)/tot_mass
    comz = np.sum(pm*pz)/tot_mass

    r2 = abs((px-comx)**2 + (py-comy)**2 + (pz-comz)**2)

    sort_r = np.argsort(r2)

    m_sort = pm[sort_r]
    r2_sort = r2[sort_r]

    half_mass = tot_mass/2.0
    cum_mass = np.cumsum(m_sort)
    idx_hmr = np.argmax(cum_mass>half_mass)

    hmr = np.sqrt(r2_sort[idx_hmr])

    rho_hm = 3*half_mass/(4*np.pi*hmr**3)

    return rho_hm

QUANTITY_REGISTRY["stellar_density"] = stellar_density
QUANTITY_TYPE["stellar_density"] = 'scalar'
QUANTITY_LABELS["stellar_density"] = r"$\rho_{\rm hm,\star}~[\rm M_\odot~pc^{-3}]$"


def stellar_ellipticity(ds):
    """
    Compute stellar ellipticity using the reduced inertia tensor method.
    Returns c/a where a, b, c are the principal axes (a >= b >= c).
    """
    if not ds.particles_exist:
        return 0.0

    ad = ds.all_data()

    stars = ad[("all", "particle_csgm")].value == 0.0

    mass = ad[('all', 'particle_mass')][stars].to('Msun').value
    px = ad[('all', 'particle_position_x')][stars].to('pc').value
    py = ad[('all', 'particle_position_y')][stars].to('pc').value
    pz = ad[('all', 'particle_position_z')][stars].to('pc').value

    return _ellipticity_from_inertial_tensor(mass, px, py, pz)

QUANTITY_REGISTRY["stellar_ellipticity"] = stellar_ellipticity
QUANTITY_TYPE["stellar_ellipticity"] = 'scalar'
QUANTITY_LABELS["stellar_ellipticity"] = r"$e_{\star}$"


def half_mass_radius(ds):
    """
    Computes the half-mass radius of the star cluster in pc.
    """
    if not ds.particles_exist:
        return 0.0

    ad = ds.all_data()

    stars = ad[("all", "particle_csgm")].value == 0.0 # star marker

    # Star positions in pc and masses in Msun
    px = ad[("all", "particle_position_x")][stars].to("pc").value
    py = ad[("all", "particle_position_y")][stars].to("pc").value
    pz = ad[("all", "particle_position_z")][stars].to("pc").value
    pm = ad[("all", "particle_mass")][stars].to("Msun").value

    tot_mass = np.sum(pm)
    comx = np.sum(pm*px)/tot_mass
    comy = np.sum(pm*py)/tot_mass
    comz = np.sum(pm*pz)/tot_mass

    r2 = abs((px-comx)**2 + (py-comy)**2 + (pz-comz)**2)

    sort_r = np.argsort(r2)

    m_sort = pm[sort_r]
    r2_sort = r2[sort_r]

    half_mass = tot_mass/2.0
    cum_mass = np.cumsum(m_sort)
    idx_hmr = np.argmax(cum_mass>half_mass)

    hmr = np.sqrt(r2_sort[idx_hmr])
    return hmr

QUANTITY_REGISTRY["half_mass_radius"] = half_mass_radius
QUANTITY_TYPE["half_mass_radius"] = 'scalar'
QUANTITY_LABELS["half_mass_radius"] = r"$R_{1/2}~[\rm{pc}]$"


def sfe(ds):
    """Computes the star formation efficiency defined as the total
    mass of stars over the total mass of gas."""
    ms = stellar_mass(ds)
    mg = gas_mass(ds)
    return ms / mg if mg > 0 else 0.0

QUANTITY_REGISTRY["sfe"] = sfe
QUANTITY_TYPE["sfe"] = 'scalar'
QUANTITY_LABELS["sfe"] = r"$\epsilon_\star$"


def sfr(ds, prev_values):
    """Compute the star formation rate in Msun/yr using output intervals as dt."""
    if prev_values == None:
        # there are no previous values; return nan
        return np.nan

    prev_time, prev_star_mass = prev_values
    sm = stellar_mass(ds)
    dm = sm - prev_star_mass
    dt = (ds.current_time.in_units('Myr').v - prev_time)*1e6 # Myr to yr

    # floor at 0. sometimes, a runaway star can make dm<0
    return max(dm/dt,0.0)

QUANTITY_REGISTRY["sfr"] = sfr
QUANTITY_TYPE["sfr"] = 'scalar'
QUANTITY_LABELS["sfr"] = r"$\rm{SFR}\,[\rm{M_\odot~yr^{-1}}]$"
QUANTITY_REQUISITES["sfr"] = ["time", "stellar_mass"]


def number_stars(ds):
    """Computes the total number of stars."""
    sm = _particle_mass_container(ds, particle_type="stars")
    return len(sm)

QUANTITY_REGISTRY["number_stars"] = number_stars
QUANTITY_TYPE["number_stars"] = 'scalar'
QUANTITY_LABELS["number_stars"] = r"$N_\star$"


def number_feedback_stars(ds):
    """Computes the total number of feedback stars given the feedback mass defined
    in the torch_user.py pointed to in the python path."""
    from torch_user import user_parameters
    p = user_parameters()
    if not ds.particles_exist:
            return 0.0

    ad = ds.all_data()

    pm = (ad[("all", "particle_old_pmass")]*yt.units.cm).to("Msun").value

    # Get particle type mask
    csgm = ad[("all", "particle_csgm")].value
    mask = (csgm == 0)

    sm = pm[mask]
    return len(sm[sm >= float(p['min_feedback_mass'])])

QUANTITY_REGISTRY["number_feedback_stars"] = number_feedback_stars
QUANTITY_TYPE["number_feedback_stars"] = 'scalar'
QUANTITY_LABELS["number_feedback_stars"] = r"$N_{\rm fb}$"


def number_sinks(ds):
    """Computes the number of sink particles."""
    sm = _particle_mass_container(ds, particle_type="sinks")
    return len(sm)

QUANTITY_REGISTRY["number_sinks"] = number_sinks
QUANTITY_TYPE["number_sinks"] = 'scalar'
QUANTITY_LABELS["number_sinks"] = r"$N_{\rm sink}$"


def max_star_mass(ds):
    """Returns the most massive star mass in Msun."""
    sm = _particle_mass_container(ds, particle_type="stars")
    return max(sm, default=0)

QUANTITY_REGISTRY["max_star_mass"] = max_star_mass
QUANTITY_TYPE["max_star_mass"] = 'scalar'
QUANTITY_LABELS["max_star_mass"] = r"$M_{\star,\rm max}\,[M_\odot]$"

def bound_star_mass_fraction(ds):
    """Computes the fraction of stellar mass that is bound over the total
    stellar mass."""
    if not ds.particles_exist:
        return 0.0

    ad = ds.all_data()
    star_idx = ad['all', 'particle_csgm'] == 0.0

    mass = ad['all','particle_mass'][star_idx].v
    pos = np.array([ad['all','particle_posx'][star_idx].v, 
                               ad['all','particle_posy'][star_idx].v, 
                               ad['all','particle_posz'][star_idx].v]).T
    vel = np.array([ad['all','particle_velx'][star_idx].v,
                                    ad['all','particle_vely'][star_idx].v,
                                    ad['all','particle_velz'][star_idx].v]).T
    # Potential field of gas+sinks at star positions
    gas_star_gpot   = mass*ds.find_field_values_at_points('gpot',pos*yt.units.cm).v
    # Potential field of stars at star positions
    star_star_gpot = mass*ds.find_field_values_at_points('bgpt', pos*yt.units.cm).v

    # Stellar kinetic energy
    v2 = np.sum(vel**2, axis=1)
    Ekin = 0.5*mass*v2

    # Total energy of stars
    Etot = Ekin + star_star_gpot + gas_star_gpot

    bound_idx = np.where(Etot < 0.0)
    bound_mass_fraction = sum(ad['all','particle_mass'][star_idx][bound_idx]).to('Msun').v
    total_mass = sum(ad['all','particle_mass'][star_idx]).to('Msun').v

    return bound_mass_fraction/total_mass

QUANTITY_REGISTRY["bound_star_mass_fraction"] = bound_star_mass_fraction
QUANTITY_TYPE["bound_star_mass_fraction"] = 'scalar'
QUANTITY_LABELS["bound_star_mass_fraction"] = r"$M_\mathrm{\star, bound}/M_\star$"

def unbound_star_ids(ds):
    """
    Returns the FLASH particle IDs of stars unbound to the system. 
    Using the exact stellar potential is an expensive calculation 
    (O(n^2)), so we use the CIC mapping of stellar potential
    from FLASH instead. 
    """
    if not ds.particles_exist:
        return []

    ad = ds.all_data()
    star_idx = ad['all', 'particle_csgm'] == 0.0

    mass = ad['all','particle_mass'][star_idx].v
    pos = np.array([ad['all','particle_posx'][star_idx].v, 
                               ad['all','particle_posy'][star_idx].v, 
                               ad['all','particle_posz'][star_idx].v]).T
    vel = np.array([ad['all','particle_velx'][star_idx].v,
                                    ad['all','particle_vely'][star_idx].v,
                                    ad['all','particle_velz'][star_idx].v]).T
    # Potential field of gas+sinks at star positions
    gas_star_gpot   = mass*ds.find_field_values_at_points('gpot',pos*yt.units.cm).v
    # Potential field of stars at star positions
    star_star_gpot = mass*ds.find_field_values_at_points('bgpt', pos*yt.units.cm).v

    # TODO: implement user option to use exact potential (sloooow) or grid potential (fast)
    if False:
        from amuse.lab import Particles
        from amuse.units import units
        stars = Particles(len(mass))
        stars.mass     = mass | units.g
        stars.position = pos | units.cm
        stars.velocity = vel | units.cm/units.s

        # Potential field of stars 
        star_star_gpot = (stars.mass*stars.potential()).value_in(units.erg) 

    # Stellar kinetic energy
    v2 = np.sum(vel**2, axis=1)
    Ekin = 0.5*mass*v2

    # Total energy of stars
    Etot = Ekin + star_star_gpot + gas_star_gpot

    # ID tags of stars
    tag = ad['all','particle_tag'][star_idx].v
    unbound_idx = np.where(Etot >= 0.0)
    runaways = tag[unbound_idx]
    n_run = len(runaways)
    print(f'I found {n_run} runaway stars!')

    return runaways

QUANTITY_REGISTRY["unbound_star_ids"] = unbound_star_ids
QUANTITY_TYPE["unbound_star_ids"] = 'vector'
QUANTITY_LABELS["unbound_star_ids"] = r"${\rm unbound~star~IDs}$"


# =============================================================================
# ROI-specific quantities
# =============================================================================

def gas_mass_roi(ds):
    """Computes the total gas mass in the user-defined region of interest."""
    region = _get_roi_region(ds)
    return _gas_mass_container(region)

QUANTITY_REGISTRY["gas_mass_roi"] = gas_mass_roi
QUANTITY_TYPE["gas_mass_roi"] = 'scalar'
QUANTITY_LABELS["gas_mass_roi"] = r"$M_\mathrm{gas, ROI}\,[M_\odot]$"


def gas_virial_ratio_roi(ds):
    """Computes the virial ratio of gas in the user-defined region of interest.
    Uses only the gas self-gravity potential (gpot); the stellar potential (bgpt)
    is excluded."""
    region = _get_roi_region(ds)
    return _gas_virial_ratio_container(region)

QUANTITY_REGISTRY["gas_virial_ratio_roi"] = gas_virial_ratio_roi
QUANTITY_TYPE["gas_virial_ratio_roi"] = 'scalar'
QUANTITY_LABELS["gas_virial_ratio_roi"] = r"$\alpha_{\rm gas}$"


def bound_gas_mass_fraction_roi(ds):
    """Computes the bound gas mass fraction in the user-defined region of interest."""
    region = _get_roi_region(ds)
    return _bound_gas_mass_fraction_container(region)

QUANTITY_REGISTRY["bound_gas_mass_fraction_roi"] = bound_gas_mass_fraction_roi
QUANTITY_TYPE["bound_gas_mass_fraction_roi"] = 'scalar'
QUANTITY_LABELS["bound_gas_mass_fraction_roi"] = r"$M_\mathrm{gas, bound}/M_{\rm tot}$"


def sfe_roi(ds):
    """Computes the SFE in the user-defined region of interest."""
    region = _get_roi_region(ds)
    mstars = stellar_mass(ds)
    mgas = gas_mass_roi(ds)
    return mstars / mgas if mgas > 0.0 else 0.0

QUANTITY_REGISTRY["sfe_roi"] = sfe_roi
QUANTITY_TYPE["sfe_roi"] = 'scalar'
QUANTITY_LABELS["sfe_roi"] = r"$\epsilon_{\star,\ \mathrm{ROI}}$"
