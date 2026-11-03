import numpy as np
from traffic_yolo.config import load_config
from traffic_yolo.pipeline import TrafficMonitorPipeline


def test_pedestrians_not_detected_as_motorcycles():
    """Verify that pure vehicle classes [2, 3, 5, 7] ignore pedestrians."""
    cfg = load_config()
    cfg["tracking"]["map_road_riders_to_motorcycle"] = False
    pipeline = TrafficMonitorPipeline(cfg)
    
    # Track classes should only contain vehicle IDs, not 0 (person)
    track_classes = list(pipeline.vehicle_classes.keys())
    if cfg["tracking"]["map_road_riders_to_motorcycle"]:
        track_classes.append(0)
        
    assert 0 not in track_classes, "Class 0 (person) must not be tracked when map_road_riders_to_motorcycle is False"
    assert set(track_classes) == {2, 3, 5, 7}, "Only Car, Moto, Bus, Truck should be tracked"
