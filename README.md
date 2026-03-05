# bear-edge
くまをAI detection
Python（OpenCV）＋pytest＋GitHub Actions
Pi上でも動かしやすく、PC上でも動画ファイルで動作確認できます。

## 前提のルール
1. I/Oは camera.py / uploader.py に閉じ込める（他は純粋関数寄り）
1. MotionDetector.process(frame) は副作用なし（stateは内部に持つが外部I/Oなし）
1. eventは events/<event_id>/ に保存し、状態はsqliteで管理
1. “送信”はスタブ可能（presigned URL or mock API）

下記構成とする
bear-edge/
  README.md
  pyproject.toml
  src/bear_edge/
    __init__.py
    config.py
    main.py
    camera.py
    motion.py
    recorder.py
    spool.py
    uploader.py
    net.py
    utils.py
  tests/
    test_motion.py
    test_spool.py
    test_recorder.py
  assets_test/
    sample_day.mp4
    sample_night.mp4
  .github/workflows/ci.yml
  docker/
    Dockerfile.dev   (任意: PC/CI用)
