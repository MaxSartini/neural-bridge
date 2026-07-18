from neural_bridge.again.configs import PHASE5_SELECTED_HEAD, PHASE7_BLOCKED, PHASE7_GROUPED


def test_frozen_endpoint_row_accounting() -> None:
    lanes = len(PHASE5_SELECTED_HEAD.run.controls)
    phase5_rows = len(PHASE5_SELECTED_HEAD.run.seeds) * lanes * (5 + 1)
    assert phase5_rows == 420

    for endpoint, expected_rows, outer_folds in (
        (PHASE7_BLOCKED, 140, 1),
        (PHASE7_GROUPED, 420, 5),
    ):
        run = endpoint.run
        member_rows = len(run.seeds) * lanes * outer_folds
        ensemble_rows = len(run.checkpoint_ensembles) * lanes * outer_folds
        assert member_rows + ensemble_rows == expected_rows
        assert all(len(group) == 3 for group in run.checkpoint_ensembles)
