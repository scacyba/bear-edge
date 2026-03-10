import numpy as np

from bear_edge.motion import MotionDetector


def make_frame(x: int = 0) -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    if x > 0:
        frame[60:180, x : x + 80] = 255
    return frame


def test_motion_detector_triggers_after_continuous_hits() -> None:
    detector = MotionDetector(min_area_px=2000, required_hits=4, window_size=5)

    outputs = []
    for i in range(8):
        frame = make_frame(x=20 + (i * 4))
        triggered, debug = detector.process(frame)
        outputs.append((triggered, debug))

    assert any(t for t, _ in outputs), "expected trigger in sequence"


def test_motion_detector_validates_input() -> None:
    detector = MotionDetector()

    bad = np.zeros((20, 20), dtype=np.uint8)
    try:
        detector.process(bad)
        assert False, "process should reject non-BGR frame"
    except ValueError:
        pass
