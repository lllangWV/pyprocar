__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
from enum import Enum
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from pyprocar.cfg import ConfigFactory, ConfigManager
from pyprocar.cfg.band_structure import BandStructureConfig
from pyprocar.cfg.base import PlotType
from pyprocar.core.atomic_orbital_index import orbital_indices
from pyprocar.core.ebs import ElectronicBandStructureMesh, ElectronicBandStructurePath
from pyprocar.core.projection import selection_resolver
from pyprocar.core.property_store import Property
from pyprocar.plotter._export import export_path
from pyprocar.plotter.bs_plot import BandStructurePlotter
from pyprocar.scripts._selection import (
    per_channel,
    projection_colorbar_title,
    projection_components,
    resolve_spins,
    signed_clim,
    take_channels,
)
from pyprocar.utils.log_utils import set_verbose_level, warn_user
from pyprocar.utils.splash import welcome

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)


class BandStructureMode(Enum):
    """
    An enumeration for defining the modes of Band Structure representations.

    Attributes
    ----------
    PLAIN : str
        Represents the band structure in a simple, where the colors are the different bands.
    PARAMETRIC : str
        Represents the band structure in a parametric form, summing over the projections.
    SCATTER : str
        Represents the band structure in a scatter plot, where the colors are the different bands.
    SACATTER : str
        Deprecated alias of SCATTER, the misspelled name earlier releases used.
    ATOMIC : str
        Represents the band structure in an atomic level plot, plots singlr kpoint bands.
    OVERLAY : str
        Represents the band structure in an overlay plot, where the colors are the selected projections
    OVERLAY_SPECIES : str
        Represents the band structure in an overlay plot, where the colors are
        the different projection of the species.
    OVERLAY_ORBITALS : str
        Represents the band structure in an overlay plot, where  the colors are
        the different projection of the orbitals.
    """

    PLAIN = "plain"
    PARAMETRIC = "parametric"
    SCATTER = "scatter"
    ATOMIC = "atomic"
    OVERLAY = "overlay"
    OVERLAY_SPECIES = "overlay_species"
    OVERLAY_ORBITALS = "overlay_orbitals"
    IPR = "ipr"
    SACATTER = "scatter"

    @classmethod
    def from_str(cls, mode: str):
        try:
            return cls(mode.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mode: {mode}. The modes available are: {cls.to_list()}"
            ) from None

    @classmethod
    def to_list(cls):
        return [mode.value for mode in cls]

    @classmethod
    def get_overlay_modes(cls):
        return [cls.OVERLAY, cls.OVERLAY_SPECIES, cls.OVERLAY_ORBITALS]


def bandsplot(
    code: str,
    dirname: str,
    mode: str = "plain",
    spins: list[int] | None = None,
    atoms: list[int] | None = None,
    orbitals: list[int] | None = None,
    items: dict | list[dict] | None = None,
    fermi: float | None = None,
    fermi_shift: float = 0,
    kticks=None,
    knames=None,
    elimit: list[float] | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    savefig: str | None = None,
    print_plot_opts: bool = False,
    export_data_file: str | Path | None = None,
    export_append_mode: bool = True,
    x_limit: list[float] | None = None,
    plot_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    parametric_kwargs: dict | None = None,
    atomic_levels_kwargs: dict | None = None,
    overlay_kwargs: dict | None = None,
    ipr_kwargs: dict | None = None,
    use_cache: bool = False,
    quiet_welcome: bool = False,
    verbose: int = 1,
    **kwargs,
):
    """A function to plot the band structutre

    Parameters
    ----------
    code : str, optional
        String to of the code used, by default "vasp"
    dirname : str, optional
        The directory name of the calculation, by default None
    mode : str, optional
        Sting for the mode of the calculation, by default "plain"
    spins : List[int], optional
        A list of spins, by default None
    atoms : List[int], optional
        A list of atoms, by default None
    orbitals : List[int], optional
        A list of orbitals, by default None
    items : dict, optional
        A dictionary where the keys are the atoms and the values a list of orbitals, by default None
    fermi : float, optional
        Float for the fermi energy, by default None. By default the fermi energy will be shifted by the fermi value that is found in the directory.
        For band structure calculations, due to convergence issues, this fermi energy might not be accurate. If so add the fermi energy from the self-consistent calculation.
    fermi_shift : float, optional
        Float to shift the fermi energy, by default 0.
    kticks : _type_, optional
        A list of kticks, by default None
    knames : _type_, optional
        A list of kanems, by default None
    elimit : List[float], optional
        A list of floats to decide the energy window, by default None
    ax : plt.Axes, optional
        A matplotlib axes, by default None
    show : bool, optional
        Boolean if to show the plot, by default True
    savefig : str, optional
        String to save the plot, by default None
    export_data_file : str or Path, optional
        The file name to export the data to. Its suffix sets the format: .csv,
        .txt, .dat or .json. If not provided the data will not be exported.
    export_append_mode : bool, optional
        Boolean to append the mode to the file name, so ``bands.dat`` becomes
        ``bands_plain.dat``. If False the data goes to ``export_data_file``.
    x_limit : List[float], optional
        The k-distance window to plot, by default None
    plot_kwargs, scatter_kwargs, parametric_kwargs : dict, optional
        Extra matplotlib keyword arguments for the plain, scatter and parametric modes.
    atomic_levels_kwargs, overlay_kwargs, ipr_kwargs : dict, optional
        Extra matplotlib keyword arguments for the atomic, overlay and ipr modes.
    print_plot_opts: bool, optional
        Boolean to print the plotting options
    quiet_welcome: bool, optional
        Boolean to not print the welcome message
    use_cache: bool, optional
        Boolean to use cache for EBS
    verbose: int, optional
        Verbosity level: 0 is silent, 1 logs progress, 2 or more adds debug logs
    **kwargs
        Plot options of :class:`pyprocar.cfg.band_structure.BandStructureConfig`, such as
        ``title``, ``cmap`` or ``x_label``. An unknown option raises ``TypeError``.

    """

    set_verbose_level(verbose)
    if quiet_welcome:
        user_logger.setLevel(logging.ERROR)

    user_logger.info("If you want more detailed logs, set verbose to 2 or more")
    user_logger.info("_" * 100)

    welcome()

    default_config = ConfigFactory.create_config(PlotType.BAND_STRUCTURE)
    config = cast(
        BandStructureConfig, ConfigManager.merge_configs(default_config, kwargs, bandsplot)
    )

    user_logger.info("_" * 100)
    if print_plot_opts:
        for key, value in default_config.as_dict().items():
            user_logger.info(f"{key} : {value}")
    user_logger.info("_" * 100)

    plot_mode = BandStructureMode.from_str(mode)
    export = None if export_data_file is None else export_path(export_data_file)

    loaded = ElectronicBandStructurePath.from_code(code, dirname, use_cache=use_cache)
    if plot_mode != BandStructureMode.ATOMIC and not isinstance(
        loaded, ElectronicBandStructurePath
    ):
        kind = (
            "a k-point mesh" if isinstance(loaded, ElectronicBandStructureMesh) else "a k-point set"
        )
        raise ValueError(
            f"{dirname} is {kind}, not a band structure along a k-path. bandsplot needs a"
            + " band-structure run (line-mode KPOINTS); plot a k-point mesh with fermi2D,"
            + " fermi3D or BandStructure2DHandler."
        )
    ebs = cast(ElectronicBandStructurePath, loaded)

    codes_with_scf_fermi = ["qe", "elk", "abinit"]
    if code in codes_with_scf_fermi and fermi is None:
        logger.info(f"No fermi given, using the found fermi energy: {ebs.fermi}")
        fermi = ebs.fermi

    if fermi is not None:
        logger.info(f"Shifting Fermi energy to zero: {fermi}")
        ebs.shift_bands(fermi_shift - fermi, inplace=True)
        y_label = r"E - E$_F$ (eV)"
    else:
        y_label = r"E (eV)"
        warn_user(
            "`fermi` is not set! Set `fermi={value}`. The plot did not shift the bands by the Fermi energy."
        )

    selection = resolve_spins(
        ebs.is_non_collinear,
        ebs.n_spin_channels,
        spins,
        plain=plot_mode == BandStructureMode.PLAIN,
    )
    channels, projection_spins = selection.channels, selection.projection_spins
    if selection.joined:
        ebs.fix_collinear_spin()
        channels = projection_spins = [0]

    if atoms is not None and isinstance(atoms[0], str):
        species = [str(name) for name in atoms]
        atoms = list(selection_resolver(ebs).resolve(species=species).atoms)
    orbitals = orbital_indices(orbitals, ebs)

    user_clim = config.clim if "clim" in kwargs else None
    projection_title = projection_colorbar_title(
        config.colorbar_title, "colorbar_title" in kwargs, ebs.is_non_collinear, projection_spins
    )

    plotter = BandStructurePlotter(ax=ax)
    n_channels = len(channels)
    style: dict[str, Any] = {
        "linestyle": per_channel(config.linestyle, n_channels),
        "alpha": per_channel(config.opacity, n_channels),
    }

    if plot_mode == BandStructureMode.ATOMIC:
        user_logger.info("Plotting bands in atomic mode")
        if ebs.n_kpoints != 1:
            raise ValueError("Atomic mode needs a single k-point calculation")
        weights = ebs.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=projection_spins)
        levels: dict[str, Any] = {
            "bands": take_channels(cast(Property, ebs.get_property("bands")), channels).value,
            "elimit": elimit,
            "scalars": take_channels(weights, channels).value,
            "cmap": config.cmap,
            "clim": user_clim or (None, None),
            **(atomic_levels_kwargs or {}),
        }
        plotter.plot_atomic_levels(**levels)
        plotter.set_colorbar_title(projection_title)
    elif plot_mode in BandStructureMode.get_overlay_modes():
        user_logger.info(f"Plotting bands in {plot_mode.value} mode")
        kind = plot_mode.value.split("_", 1)[1] if "_" in plot_mode.value else "items"
        weights = projection_components(
            ebs, kind, atoms=atoms, orbitals=orbitals, items=items, spins=projection_spins
        )
        plotter.plot_overlay(
            ebs.kpath,
            take_channels(cast(Property, ebs.bands), channels).value,
            weights=[take_channels(w, channels).value for w in weights],
            labels=[str(w.label) for w in weights],
            **(overlay_kwargs or {}),
        )
    else:
        bands = take_channels(cast(Property, ebs.bands), channels)
        if plot_mode == BandStructureMode.PLAIN:
            user_logger.info("Plotting bands in plain mode")
            line_style: dict[str, Any] = {
                "color": per_channel(
                    config.spin_colors if n_channels > 1 else config.color, n_channels
                ),
                "linewidth": per_channel(config.linewidth, n_channels),
                **style,
                **(plot_kwargs or {}),
            }
            plotter.plot(bands, **line_style)
        else:
            if plot_mode == BandStructureMode.IPR:
                user_logger.info("Plotting bands in IPR mode")
                scalars = ebs.compute_ebs_ipr()
                colorbar_title = "Inverse Participation Ratio"
                artist_kwargs: dict[str, Any] = {"collection_kwargs": ipr_kwargs}
            else:
                user_logger.info(f"Plotting bands in {plot_mode.value} mode")
                scalars = ebs.compute_projected_sum(
                    atoms=atoms, orbitals=orbitals, spins=projection_spins
                )
                colorbar_title = projection_title
                artist_kwargs = {
                    "collection_kwargs": parametric_kwargs,
                    "scatter_kwargs": scatter_kwargs,
                }
            scalars = take_channels(scalars, channels)
            plotter.plot(
                bands,
                scalars_data=scalars,
                scalars_mode="scatter" if plot_mode == BandStructureMode.SCATTER else "parametric",
                scalars_cmap=config.cmap,
                scalars_clim=user_clim
                or signed_clim(scalars)
                or (0.0, float(np.nanmax(scalars.value)) or 1.0),
                **artist_kwargs,
                **style,
            )
            plotter.set_colorbar_label(colorbar_title)

    if kticks is not None or knames is not None:
        plotter.set_xticks(kticks, knames)
    plotter.set_yticks(interval=elimit)
    if x_limit is not None:
        plotter.set_xlim(x_limit)
    if elimit is not None:
        plotter.set_ylim(elimit)
    plotter.set_ylabel(label=y_label)
    plotter.set_xlabel(label=config.x_label)

    if fermi is not None:
        plotter.draw_fermi(
            fermi_level=fermi_shift,
            color=config.fermi_color,
            linestyle=config.fermi_linestyle,
            linewidth=config.fermi_linewidth,
        )

    if config.title is not None:
        plotter.set_title(config.title)
    else:
        plotter.set_title()
    plotter.grid()

    if savefig is not None:
        plotter.save(savefig)
    if show:
        plotter.show()

    if export is not None:
        if export_append_mode:
            export = export.with_name(f"{export.stem}_{plot_mode.value}{export.suffix}")
        plotter.export_data(export)

    return plotter.fig, plotter.ax
