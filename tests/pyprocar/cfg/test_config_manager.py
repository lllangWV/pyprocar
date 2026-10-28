import pytest

from pyprocar.cfg import ConfigManager, PlotType
from pyprocar.cfg.band_structure import BandStructureConfig
from pyprocar.cfg.unfold import UnfoldingConfig
from tests.utils import ROOT_DIR


def _plot(use_cache: bool = False, **kwargs):
    """A one-call function that takes plot options as ``**kwargs``."""


def test_merge_configs_leaves_default_untouched():
    default = BandStructureConfig(plot_type=PlotType.BAND_STRUCTURE)

    merged = ConfigManager.merge_configs(default, {"fermi_linewidth": 7.5}, _plot)

    assert isinstance(merged, BandStructureConfig)
    assert merged.fermi_linewidth == 7.5
    assert default.fermi_linewidth == 1


def test_merge_configs_rejects_an_unknown_option_naming_the_caller_and_the_closest_name():
    default = BandStructureConfig(plot_type=PlotType.BAND_STRUCTURE)

    with pytest.raises(TypeError) as error:
        ConfigManager.merge_configs(default, {"xlabel": "k", "cache": False}, _plot)

    assert str(error.value) == (
        "_plot() got unexpected keyword arguments 'xlabel' (did you mean 'x_label'?),"
        + " 'cache' (did you mean 'use_cache'?); print_plot_opts=True lists its plot options"
    )


def test_merge_configs_rejects_an_option_close_to_nothing_without_a_guess():
    default = BandStructureConfig(plot_type=PlotType.BAND_STRUCTURE)

    with pytest.raises(TypeError, match=r"argument 'zzz';"):
        ConfigManager.merge_configs(default, {"zzz": 1}, _plot)


def test_merge_configs_rejects_plot_type_which_the_factory_sets():
    default = BandStructureConfig(plot_type=PlotType.BAND_STRUCTURE)

    with pytest.raises(TypeError, match="'plot_type'"):
        ConfigManager.merge_configs(default, {"plot_type": PlotType.UNFOLD}, _plot)


def test_merge_configs_skips_the_options_the_caller_reads_itself():
    default = BandStructureConfig(plot_type=PlotType.BAND_STRUCTURE)

    merged = ConfigManager.merge_configs(
        default, {"off_screen": True, "linewidth": (2.0, 2.0)}, _plot, also_accepts=("off_screen",)
    )

    assert isinstance(merged, BandStructureConfig)
    assert merged.linewidth == (2.0, 2.0)
    assert not hasattr(merged, "off_screen")


def test_unfold_config_has_no_options_without_a_reader():
    options = UnfoldingConfig(plot_type=PlotType.UNFOLD).as_dict()

    assert {"modes", "weighted_color", "weighted_width"} & options.keys() == set()
    assert options["cmap"] == "jet"
    assert not (ROOT_DIR / "pyprocar" / "cfg" / "unfold.yml").exists()
