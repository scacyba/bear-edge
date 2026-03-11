from __future__ import annotations

from collections import deque
from typing import Deque

import cv2
import numpy as np


class MotionDetector:
    """Lightweight motion detector for edge devices.

    Detection logic combines frame differencing and MOG2 foreground segmentation,
    then applies an area threshold and temporal continuity check.
    """

    def __init__(
        self,
        min_area_px: int = 5000,
        required_hits: int = 8,
        window_size: int = 10,
        diff_threshold: int = 25,
        mog2_history: int = 300,
        mog2_var_threshold: int = 16,
        detect_shadows: bool = False,
    ) -> None:
        if min_area_px <= 0:
            raise ValueError("min_area_px must be > 0")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if required_hits <= 0:
            raise ValueError("required_hits must be > 0")
        if required_hits > window_size:
            raise ValueError("required_hits must be <= window_size")

        self.min_area_px = min_area_px
        self.required_hits = required_hits
        self.window_size = window_size
        self.diff_threshold = diff_threshold

        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history,
            varThreshold=mog2_var_threshold,
            detectShadows=detect_shadows,
        )
        self._prev_gray: np.ndarray | None = None
        self._recent_hits: Deque[bool] = deque(maxlen=window_size)

        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def process(self, frame: np.ndarray) -> tuple[bool, dict]:
        """Process a BGR frame and return trigger status and debug info."""
        if frame.dtype != np.uint8:
            raise ValueError("frame must be np.uint8")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be BGR (H, W, 3)")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fg_mog2 = self._mog2.apply(frame)
        _, fg_mog2 = cv2.threshold(fg_mog2, 200, 255, cv2.THRESH_BINARY)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._recent_hits.append(False)
            return False, {
                "initialized": False,
                "motion_area_px": 0,
                "area_threshold_px": self.min_area_px,
                "frame_motion": False,
                "recent_hits": list(self._recent_hits),
                "hits_in_window": 0,
                "required_hits": self.required_hits,
                "window_size": self.window_size,
            }

        diff = cv2.absdiff(gray, self._prev_gray)
        _, fg_diff = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # and‚¾‚Æ”’•‚Å‚Í•ß‚Ü‚ç‚È‚¢B
        #combined = cv2.bitwise_and(fg_diff, fg_mog2)
        combined = cv2.bitwise_or(fg_diff, fg_mog2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self._kernel, iterations=1)
        combined = cv2.dilate(combined, self._kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)

        motion_area_px = 0
        boxes = []
        for i in range(1, num_labels):
            component_area = int(stats[i, cv2.CC_STAT_AREA])
            if component_area >= self.min_area_px:
                motion_area_px += component_area

                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                boxes.append((x, y, w, h))

        frame_motion = motion_area_px >= self.min_area_px
        self._recent_hits.append(frame_motion)

        hits_in_window = int(sum(self._recent_hits))
        triggered = hits_in_window >= self.required_hits

        self._prev_gray = gray

        debug = {
            "initialized": True,
            "motion_area_px": motion_area_px,
            "boxes": boxes,
            "area_threshold_px": self.min_area_px,
            "frame_motion": frame_motion,
            "recent_hits": list(self._recent_hits),
            "hits_in_window": hits_in_window,
            "required_hits": self.required_hits,
            "window_size": self.window_size,
        }
        return triggered, debug
