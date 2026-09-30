# data/

Not in git. What goes here:

- `frames/` — camera frames from the robot used by the tests in `eval/`
  (`pose1.jpg` … `pose3.jpg`: someone facing the camera; `scene.jpg`,
  `quiet2.jpg`: nobody engaging; `real_board.jpg`: a handwritten board).
  Any frames of your own with the same names will do.
- `models/` — face detection and recognition for `face_compare.py`, from the
  OpenCV model zoo: `face_detection_yunet_2023mar.onnx` and
  `face_recognition_sface_2021dec.onnx`.
- `lfw/` — Labeled Faces in the Wild, only for `eval/test_face_compare.py`.
- `live_logs/` — one file per live session, written by `john/one_mind.py`.
- `words.txt` — optional, one English word per line (e.g. the `words` file of
  a Unix system). Anagrams only use its lowercase entries, which keeps out
  abbreviations and names; without it they use word frequencies alone.
- `stockfish/` — optional, a Stockfish binary for chess (any file named
  `stockfish*` below it is used), e.g. the official Linux build.
