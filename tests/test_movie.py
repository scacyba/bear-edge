import cv2
from pathlib import Path
from bear_edge.motion import MotionDetector
import boto3
import datetime

VIDEO_PATH = "bear_towada.mp4"
OUTPUT_DIR = Path("detected_frames")

#COOLDOWN_FRAMES = 90  # 3秒 (30fps想定)
COOLDOWN_FRAMES = 15

BUCKET = "bear-camera-images"
s3 = boto3.client("s3")

def lambda_stub(frame, frame_id):
    """
    Lambdaの代わりのスタブ。デバッグ用
    検知フレームを保存
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    filename = OUTPUT_DIR / f"detected_{frame_id:05d}.jpg"
    cv2.imwrite(str(filename), frame)

    print(f"saved {filename}")


def s3upload(frame):

    # OpenCV frame → JPEGエンコード
    success, buffer = cv2.imencode(".jpg", frame)

    if not success:
        raise Exception("JPEG encode failed")

    # S3 key (ファイル名)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    key = f"camera01/{timestamp}.jpg"

    # S3アップロード
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=buffer.tobytes(),
        ContentType="image/jpeg"
    )

    print("uploaded:", key)

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
            # lambda_stub(roi, frame_id)
            s3upload(frame)
            last_trigger_frame = frame_id

        # 1秒 (30fps想定)ごとにデバッグ
        if frame_id % 30 == 0:
            print(debug)

        frame_id += 1

    cap.release()


if __name__ == "__main__":
    main()