from pathlib import Path

import pytest

from pyprocar.core.dos import DensityOfStates
from pyprocar.core.ebs import ElectronicBandStructurePath
from tests.utils.user_warning import user_warning

pytestmark = pytest.mark.guards_existing_behaviour(
    reason="pins from_code dispatch while the parser lookup moves out of core"
)


@pytest.mark.parametrize("cls", [ElectronicBandStructurePath, DensityOfStates])
def test_from_code_rejects_an_unknown_code(cls: type, tmp_path: Path):
    with pytest.raises(ValueError, match=r"Invalid code: nope\. Valid codes are: \n    lobster\n"):
        cls.from_code("nope", str(tmp_path))


def test_ebs_from_code_raises_when_the_parser_finds_no_ebs(tmp_path: Path):
    with (
        pytest.raises(ValueError, match="The frmsf parser found no ebs in"),
        user_warning(__file__, "FrmSrf (file not found|extractor available)"),
    ):
        ElectronicBandStructurePath.from_code("frmsf", str(tmp_path))


def test_dos_from_code_raises_when_the_parser_finds_no_dos(tmp_path: Path):
    with (
        pytest.raises(ValueError, match="The bxsf parser found no dos in"),
        user_warning(__file__, "No in.bxsf, \\*.bxsf or \\*_BXSF file found"),
    ):
        DensityOfStates.from_code("bxsf", str(tmp_path))
