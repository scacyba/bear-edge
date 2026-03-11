import cv2
from pathlib import Path
from bear_edge.motion import MotionDetector


VIDEO_PATH = "bear_towada.mp4"
OUTPUT_DIR = Path("detected_frames")

#COOLDOWN_FRAMES = 90  # 3秒 (30fps想定)
COOLDOWN_FRAMES = 15

def lambda_stub(frame, frame_id):
    """
    Lambdaの代わり
    検知フレームを保存
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    filename = OUTPUT_DIR / f"detected_{frame_id:05d}.jpg"
    cv2.imwrite(str(filename), frame)

    print(f"saved {filename}")


def main():

    detector = MotionDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("video open failed")

    frame_id = 0
    last_trigger_frame = -COOLDOWN_FRAMES

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        triggered, debug = detector.process(frame)

        boxes = debug.get("boxes", [])


        # クールダウン判定
        if triggered and (frame_id - last_trigger_frame) >= COOLDOWN_FRAMES:
            for (x, y, w, h) in boxes:
                roi = frame[y:y+h, x:x+w]
            lambda_stub(roi, frame_id)
            last_trigger_frame = frame_id

        # 1秒 (30fps想定)ごとにデバッグ
        if frame_id % 30 == 0:
            print(debug)

        frame_id += 1

    cap.release()


if __name__ == "__main__":
    main()