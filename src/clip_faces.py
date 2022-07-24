import os
import argparse
from typing import Any
import cv2


def main():
    # 引数をパースする
    parser = argparse.ArgumentParser(
        "generate aligned face images from an image")
    parser.add_argument("image", help="input image file path (./image.jpg)")
    args = parser.parse_args()

    # 引数から画像ファイルのパスを取得
    path = args.image
    directory = os.path.dirname(__file__)
    # directory = os.path.dirname(args.image)
    # if not directory:
    #     directory = os.path.dirname(__file__)
    #     path = os.path.join(directory, args.image)

    # 画像を開く
    image = cv2.imread(path)
    if image is None:
        exit()

    # 画像が3チャンネル以外の場合は3チャンネルに変換する
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # モデルを読み込む
    weights = os.path.join(directory, "..", "model", "yunet.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    weights = os.path.join(directory, "..", "model",
                           "face_recognizer_fast.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    # 入力サイズを指定する
    height, width, _ = image.shape
    face_detector.setInputSize((width, height))

    # 顔を検出する
    _, faces = face_detector.detect(image)

    # 検出された顔を切り抜く
    aligned_faces = []
    if faces is not None:
        for face in faces:
            aligned_face: Any = face_recognizer.alignCrop(image, face)
            aligned_faces.append(aligned_face)

    # 画像を表示、保存する
    for i, aligned_face in enumerate(aligned_faces):
        cv2.imwrite(os.path.join(
            directory, "out", "img", f"face{i}.jpg"), aligned_face)


if __name__ == '__main__':
    main()
