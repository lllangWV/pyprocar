import copy
import difflib
import inspect
from collections.abc import Collection, Mapping
from dataclasses import fields
from types import FunctionType, MethodType
from typing import Any

from pyprocar.cfg.band_structure import BandStructureConfig
from pyprocar.cfg.band_structure_2d import Bandstructure2DConfig
from pyprocar.cfg.base import BaseConfig, PlotType
from pyprocar.cfg.dos import DensityOfStatesConfig
from pyprocar.cfg.fermi_surface_2d import FermiSurface2DConfig
from pyprocar.cfg.fermi_surface_3d import FermiSurface3DConfig
from pyprocar.cfg.unfold import UnfoldingConfig


class ConfigFactory:
    """
    Factory class for creating configuration objects based on the plot type.

    This class provides a static method to create the appropriate configuration
    object for different types of plots (Fermi surface, band structure, density of states).

    Methods
    -------
    create_config(plot_type: PlotType, **kwargs) -> BaseConfig
        Creates and returns a configuration object based on the specified
    Parameters
    ----------
    plot_type : PlotType
        The type of plot for which to create a configuration object.
    **kwargs : dict
        Additional keyword arguments to pass to the configuration object construct
    Returns
    -------
    BaseConfig
        A configuration object of the appropriate type for the specified plo
    Raises
    ------
    ValueError
        If an unknown plot type is specified.
    """

    @staticmethod
    def create_config(plot_type: PlotType, **kwargs):
        if plot_type == PlotType.FERMI_SURFACE_3D:
            return FermiSurface3DConfig(plot_type=plot_type, **kwargs)
        elif plot_type == PlotType.BAND_STRUCTURE:
            return BandStructureConfig(plot_type=plot_type, **kwargs)
        elif plot_type == PlotType.DENSITY_OF_STATES:
            return DensityOfStatesConfig(plot_type=plot_type, **kwargs)
        elif plot_type == PlotType.UNFOLD:
            return UnfoldingConfig(plot_type=plot_type, **kwargs)
        elif plot_type == PlotType.BAND_STRUCTURE_2D:
            return Bandstructure2DConfig(plot_type=plot_type, **kwargs)
        elif plot_type == PlotType.FERMI_SURFACE_2D:
            return FermiSurface2DConfig(plot_type=plot_type, **kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")


class ConfigManager:
    """
    Manager class for handling configuration objects.

    This class merges user-provided configuration settings with default configurations.

    Methods
    -------
    merge_configs(default_config, user_config, caller, also_accepts=()) -> BaseConfig
        Returns a copy of `default_config` updated with settings from `user_config`.
        `default_config` itself is left unchanged.

    Parameters
    ----------
    default_config : BaseConfig
        The default configuration object. It is not modified.
    user_config : Mapping[str, Any]
        The plot options the user passed to `caller` as keyword arguments.
    caller : function or method
        The function that took `user_config` as ``**kwargs``. The error names it.
    also_accepts : Collection[str]
        Keys that `caller` reads from ``**kwargs`` itself. They are not set on the config.

    Returns
    -------
    BaseConfig
        A new configuration object with settings from both default and user configurations.

    Raises
    ------
    TypeError
        If a key of `user_config` is neither a field of `default_config` nor in
        `also_accepts`, as Python raises for an unexpected keyword argument.
    """

    @staticmethod
    def merge_configs(
        default_config: BaseConfig,
        user_config: Mapping[str, Any],
        caller: FunctionType | MethodType,
        also_accepts: Collection[str] = (),
    ) -> BaseConfig:
        options = {f.name for f in fields(default_config)} - {"plot_type"}
        unknown = [key for key in user_config if key not in options and key not in also_accepts]
        if unknown:
            raise TypeError(_unexpected_keywords(caller, unknown, options | set(also_accepts)))
        config = copy.deepcopy(default_config)
        for key, value in user_config.items():
            if key in options:
                setattr(config, key, value)
        return config

    @staticmethod
    def merge_config(default_config: BaseConfig, attribute: str, value: Any):
        if hasattr(default_config, attribute):
            setattr(default_config, attribute, value)
        else:
            raise AttributeError(f"{attribute} is not a valid configuration option.")
        return default_config


def _unexpected_keywords(
    caller: FunctionType | MethodType, unknown: list[str], options: set[str]
) -> str:
    """Name each unknown keyword of `caller` with the closest valid name."""
    parameters = inspect.signature(caller).parameters.values()
    valid = sorted(
        options | {p.name for p in parameters if p.kind is not inspect.Parameter.VAR_KEYWORD}
    )
    names = []
    for key in unknown:
        close = difflib.get_close_matches(key, valid, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        names.append(f"{key!r}{hint}")
    plural = "s" if len(unknown) > 1 else ""
    return (
        f"{caller.__qualname__}() got unexpected keyword argument{plural} {', '.join(names)};"
        + " print_plot_opts=True lists its plot options"
    )
