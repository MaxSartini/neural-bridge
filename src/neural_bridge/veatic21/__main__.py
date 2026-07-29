"""Command-line entry point for fresh VEATIC 2.1 work."""

from __future__ import annotations

import argparse
import json

from neural_bridge.veatic21.phase00 import run_phase00, verify_phase00_output
from neural_bridge.veatic21.phase01 import run_phase01, verify_phase01_output
from neural_bridge.veatic21.phase02_registration import (
    run_phase02_registration,
    verify_phase02_registration,
)
from neural_bridge.veatic21.phase02_stage_a import run_phase02_stage_a
from neural_bridge.veatic21.phase02_stage_a_backtest import run_phase02_executor_backtest
from neural_bridge.veatic21.phase02_stage_a_rescue_backtest import (
    register_rescue_executor_backtest,
    run_rescue_executor_backtest,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_registration import (
    run_phase02_stage_a_rescue_registration,
    verify_phase02_stage_a_rescue_registration,
)
from neural_bridge.veatic21.phase02_stage_a_saturated import run_phase02_stage_a_saturated
from neural_bridge.veatic21.phase02_stage_a_saturated_verify import (
    verify_phase02_stage_a_saturated_output,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh VEATIC 2.1 scientific runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("phase00", help="run the complete fresh Phase 00 audit")
    subparsers.add_parser("verify-phase00", help="verify the sealed Phase 00 outputs")
    subparsers.add_parser("phase01", help="run fresh Phase 01 label alignment")
    subparsers.add_parser("verify-phase01", help="verify the sealed Phase 01 outputs")
    subparsers.add_parser("register-phase02", help="freeze the fresh Phase 02 AR experiment")
    subparsers.add_parser("verify-phase02-registration", help="verify the Phase 02 AR freeze")
    stage_a = subparsers.add_parser("phase02-stage-a", help="run/resume inner Stage A screen")
    stage_a.add_argument("--max-units", type=int, default=None)
    subparsers.add_parser(
        "phase02-executor-backtest",
        help="run/resume the registered hardware-saturation backtest",
    )
    subparsers.add_parser(
        "phase02-stage-a-saturated",
        help="run/resume the full Stage A matrix with the frozen selected executor",
    )
    subparsers.add_parser(
        "verify-phase02-stage-a-saturated",
        help="exhaustively verify every saturated Stage A output and ledger entry",
    )
    subparsers.add_parser(
        "register-phase02-stage-a-rescue",
        help="freeze the exact Stage A undertrained-cell rescue registry",
    )
    subparsers.add_parser(
        "verify-phase02-stage-a-rescue-registration",
        help="independently re-derive and verify the Stage A rescue registry",
    )
    subparsers.add_parser(
        "register-phase02-stage-a-rescue-executor-backtest",
        help="freeze the sparse rescue systems-backtest matrix",
    )
    subparsers.add_parser(
        "phase02-stage-a-rescue-executor-backtest",
        help="run/resume the registered sparse rescue systems backtest",
    )
    arguments = parser.parse_args()

    if arguments.command == "phase00":
        result = run_phase00()
    elif arguments.command == "phase01":
        result = run_phase01()
    elif arguments.command == "verify-phase01":
        result = verify_phase01_output()
    elif arguments.command == "register-phase02":
        result = run_phase02_registration()
    elif arguments.command == "verify-phase02-registration":
        result = verify_phase02_registration()
    elif arguments.command == "phase02-stage-a":
        result = run_phase02_stage_a(max_units=arguments.max_units)
    elif arguments.command == "phase02-executor-backtest":
        result = run_phase02_executor_backtest()
    elif arguments.command == "phase02-stage-a-saturated":
        result = run_phase02_stage_a_saturated()
    elif arguments.command == "verify-phase02-stage-a-saturated":
        result = verify_phase02_stage_a_saturated_output()
    elif arguments.command == "register-phase02-stage-a-rescue":
        result = run_phase02_stage_a_rescue_registration()
    elif arguments.command == "verify-phase02-stage-a-rescue-registration":
        result = verify_phase02_stage_a_rescue_registration()
    elif arguments.command == "register-phase02-stage-a-rescue-executor-backtest":
        result = register_rescue_executor_backtest()
    elif arguments.command == "phase02-stage-a-rescue-executor-backtest":
        result = run_rescue_executor_backtest()
    else:
        result = verify_phase00_output()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
