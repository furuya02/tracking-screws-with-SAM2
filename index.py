import os
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from annotator import Annotator

# from pointer import Pointer
from bounding_box import BoundingBox

# from mask_util import MaskUtil
from util import (
    init_dir,
    get_video_specifications,
    create_frames_using_ffmpeg,
    allow_tf32,
)

# 定数定義
SAM2_HOME = "/segment-anything-2"
HOME = "/home"
VIDEO_FULL_PATH = f"{HOME}/screw.mp4"
SCALE = 0.8  # 1920 * 1080 => 1536 * 864
FPS = 1  # 切り取りフレームのFPS
CHECKPOINT = f"{SAM2_HOME}/checkpoints/sam2_hiera_tiny.pt"
CONFIG = "sam2_hiera_t.yaml"
OUTPUT_PATH = f"{HOME}/output"
BASE_NAME = os.path.basename(VIDEO_FULL_PATH).split(".")[0]
SOURCE_PATH = f"{OUTPUT_PATH}/{BASE_NAME}"
FRAME_PATH = f"{SOURCE_PATH}/frame"
ANNOTATION_PATH = f"{SOURCE_PATH}/annotation"
PNG_PATH = f"{SOURCE_PATH}/png"
TXT_PATH = f"{SOURCE_PATH}/txt"


def print_paths():
    print(f"VIDEO_FULL_PATH:{VIDEO_FULL_PATH}")
    print(f"OUTPUT_PATH:{OUTPUT_PATH} SOURCE_PATH:{SOURCE_PATH}")
    print(f"CHECKPOINT:{CHECKPOINT} CONFIG:{CONFIG}")


def get_video_specs(video_path):
    return get_video_specifications(video_path)


def initialize_directories():
    init_dir(OUTPUT_PATH, SOURCE_PATH, FRAME_PATH, PNG_PATH, TXT_PATH, ANNOTATION_PATH)


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    # return device


def process_frame(frame_idx, object_ids, mask_logits, source_frames, annotator):
    frame_path = source_frames[frame_idx]
    frame = cv2.imread(frame_path)
    masks = (mask_logits > 0.0).cpu().numpy()
    masks = np.squeeze(masks).astype(bool)

    # 検出領域をマスクした画像を表示（動作確認用）
    annotated_frame = annotator.set_mask(frame, masks, object_ids)
    cv2.imwrite(f"{ANNOTATION_PATH}/{frame_idx:05}.png", annotated_frame)
    cv2.imshow("annotated_frame", annotated_frame)
    if frame_idx == 0:
        cv2.moveWindow("annotated_frame", 500, 200)
    cv2.waitKey(1)


def main():
    # パス情報を表示
    print_paths()

    # VIDEOの諸元を取得
    VIDEO_COUNT, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS = get_video_specs(VIDEO_FULL_PATH)
    print(
        f"VIDEO_COUNT:{VIDEO_COUNT} VIDEO_WIDTH:{VIDEO_WIDTH} VIDEO_HEIGHT:{VIDEO_HEIGHT} VIDEO_FPS:{VIDEO_FPS}"
    )

    # 作成するフレームの諸元を計算
    FRAME_WIDTH = int(VIDEO_WIDTH * SCALE)
    FRAME_HEIGHT = int(VIDEO_HEIGHT * SCALE)
    print(
        f"FRAME_WIDTH:{FRAME_WIDTH} FRAME_HEIGHT:{FRAME_HEIGHT} FRAME_SKIP:{VIDEO_FPS / FPS}"
    )

    # 作業ディレクトリの初期化
    initialize_directories()

    # デバイスの設定
    # device = setup_device()
    setup_device()

    # TensorFloat-32 テンソル コアを使用できる場所を制御する
    allow_tf32()

    # バウンディングボックスの取得用クラス
    bounding_box = BoundingBox()
    # マスク画像の生成用クラス
    annotator = Annotator()

    ################################################################
    # 画像の生成と対象のバウンディングボックスの取得
    ################################################################
    # JPEG画像の生成
    source_frames = create_frames_using_ffmpeg(
        VIDEO_FULL_PATH, FRAME_PATH, FRAME_WIDTH, FRAME_HEIGHT, FPS
    )
    # ターゲットのバウンディングボックスの取得
    target_frame_idx = 0  # ポイントを指定するフレーム
    target_object_id = 0  # ターゲットは、1つのオブジェクトのみ
    target_box = bounding_box.get_box(source_frames[target_frame_idx])

    ################################################################
    # モデルの初期化
    ################################################################

    # SAM2 モデルのロード
    sam2_model = build_sam2_video_predictor(CONFIG, CHECKPOINT)

    # 推論用stateの初期化
    inference_state = sam2_model.init_state(video_path=FRAME_PATH)
    sam2_model.reset_state(inference_state)

    # プロンプト追加(ボックスで指定する場合)
    _, object_ids, mask_logits = sam2_model.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=target_frame_idx,
        obj_id=target_object_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=target_box,
    )
    # プロンプト追加(ポイントで指定する場合)
    # _, object_ids, mask_logits = sam2_model.add_new_points(
    #     inference_state=inference_state,
    #     frame_idx=target_frame_idx,
    #     obj_id=object_id,
    #     points=input_point, # np.array([[550, 230], [561, 342]], dtype=np.float32)
    #     labels=input_label, # np.array([1, 1], np.int32)  # 1: 前景点、0:背景点
    # )

    ################################################################
    # 推論
    ################################################################
    for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(
        inference_state
    ):
        frame_path = source_frames[frame_idx]
        frame = cv2.imread(frame_path)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)

        ################################################################
        # 検出領域をマスクした画像を表示（動作確認用）
        ################################################################
        annotated_frame = annotator.set_mask(frame, masks, object_ids)

        cv2.imwrite(f"{ANNOTATION_PATH}/{frame_idx:05}.png", annotated_frame)
        cv2.imshow("annotated_frame", annotated_frame)
        if frame_idx == 0:
            cv2.moveWindow("annotated_frame", 500, 200)

        cv2.waitKey(1)

    print("Done")


if __name__ == "__main__":
    main()
