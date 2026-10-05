"""Unit tests for temporal consensus use case."""
from shelf_monitor.entities.common import AlertType, CellState
from shelf_monitor.usecases.temporal_consensus import TemporalConsensusManager


def test_temporal_consensus_n_confirm():
    # Requires 3 consecutive frames before raising alert
    mgr = TemporalConsensusManager(n_confirm=3, n_clear=2)

    cell_update = [
        {"row_idx": 0, "item_idx": 0, "state": CellState.EMPTY, "expected_sku": "sku_A"}
    ]

    # Frame 1: no alert yet
    res1 = mgr.update_cell_states("shelf_1", cell_update)
    assert len(res1["new_alerts"]) == 0

    # Frame 2: no alert yet
    res2 = mgr.update_cell_states("shelf_1", cell_update)
    assert len(res2["new_alerts"]) == 0

    # Frame 3: reaches threshold 3 -> raises alert!
    res3 = mgr.update_cell_states("shelf_1", cell_update)
    assert len(res3["new_alerts"]) == 1
    assert res3["new_alerts"][0]["alert_type"] == AlertType.OOS


def test_temporal_consensus_clear_alert():
    mgr = TemporalConsensusManager(n_confirm=2, n_clear=2)

    # Frame 1 & 2: EMPTY -> Alert
    mgr.update_cell_states("shelf_1", [{"row_idx": 0, "item_idx": 0, "state": CellState.EMPTY, "expected_sku": "sku_A"}])
    res2 = mgr.update_cell_states("shelf_1", [{"row_idx": 0, "item_idx": 0, "state": CellState.EMPTY, "expected_sku": "sku_A"}])
    assert len(res2["new_alerts"]) == 1

    # Frame 3: OK -> streak starts
    res3 = mgr.update_cell_states("shelf_1", [{"row_idx": 0, "item_idx": 0, "state": CellState.OK, "expected_sku": "sku_A"}])
    assert len(res3["cleared_alerts"]) == 0

    # Frame 4: OK -> streak reaches n_clear=2 -> alert cleared!
    res4 = mgr.update_cell_states("shelf_1", [{"row_idx": 0, "item_idx": 0, "state": CellState.OK, "expected_sku": "sku_A"}])
    assert len(res4["cleared_alerts"]) == 1
