import logging
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
ROOT_TEST_DIR = CURRENT_DIR.parent
ROOT_DIR = ROOT_TEST_DIR.parent

sys.path.append(str(ROOT_DIR))

from pyprocar import io
from tests.utils.data import ALL_TEST_CASES, DOS_CALC_TYPES, EBS_CALC_TYPES

user_logger = logging.getLogger("user")
failed: list[str] = []

for calc_test_case in ALL_TEST_CASES:
    path = calc_test_case.path

    mat_system = calc_test_case.mat_system
    code = calc_test_case.code
    version = calc_test_case.version
    mag_type = calc_test_case.mag_type
    calc_type = calc_test_case.calc_type

    parser = io.get_parser(calc_test_case.code, calc_test_case.path)

    # A parser returns None when the files hold no DOS or bands, and raises a
    # ValueError when they cannot be read correctly; neither writes a reference.
    if calc_type in DOS_CALC_TYPES:
        attr = "dos"
    elif calc_type in EBS_CALC_TYPES:
        attr = "ebs"
    else:
        continue
    try:
        result = getattr(parser, attr)
    except ValueError as exc:
        failed.append(f"{calc_test_case.get_id()}: {exc}")
        continue
    if result is None:
        user_logger.warning(f"No {attr.upper()} found for {calc_test_case.get_id()}")
        continue
    result.save(path / f"{attr}.pkl")

if failed:
    sys.exit("Could not read:\n" + "\n".join(failed))
