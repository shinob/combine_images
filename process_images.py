#!/usr/bin/python3

import cv2
import numpy as np
import glob
import os
from datetime import date

# --- 設定 ---
# align_and_crop.py の設定
OUTPUT_WIDTH = 5700
OUTPUT_HEIGHT = 3900
# TEMPLATE_FILENAME = 'target2.jpg' # 動的に生成するため不要

# combine_images.py の設定
OUTPUT_FILENAME = f'{date.today().isoformat()}.jpg'
# --- 設定ここまで ---


def find_template_location(image, template):
    """
    画像の中からテンプレートの最も一致する場所（左上の座標）を見つける
    """
    # テンプレートマッチングを実行
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # 最も一致する場所の座標を取得
    _, _, _, max_loc = cv2.minMaxLoc(result)
    return max_loc


def crop_center(img, crop_width, crop_height):
    """
    画像の中央を切り出す
    """
    img_height, img_width, _ = img.shape
    start_x = (img_width - crop_width) // 2
    start_y = (img_height - crop_height) // 2
    return img[start_y:start_y + crop_height, start_x:start_x + crop_width]


if __name__ == '__main__':
    # --- align_and_crop.py の処理 ---
    image_files = sorted(glob.glob('IMG_*.JPG'))
    if not image_files:
        print("処理対象のIMG_*.JPGファイルが見つかりません。")
        exit()

    try:
        # 最初の画像をテンプレート生成元として使用
        template_source_filename = image_files[0]
        print(f"テンプレートを '{template_source_filename}' から生成します。")
        source_for_template = cv2.imread(template_source_filename)
        if source_for_template is None:
            raise cv2.error(f"画像の読み込みに失敗しました: {template_source_filename}")

        # 指定された座標とサイズで切り出してテンプレートを作成
        crop_x, crop_y, crop_w, crop_h = 1600, 1500, 1000, 1000
        template_img = source_for_template[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        print(f"  - 座標 ({crop_x}, {crop_y}) から 幅{crop_w}x高さ{crop_h} で切り出しました。")

    except Exception as e:
        print(f"エラー: テンプレートの生成に失敗しました。 - {e}")
        exit()

    match_locations = []
    images = {}
    print("\n各画像からテンプレートの位置を検出しています...")
    for filename in image_files:
        source_img = cv2.imread(filename)
        if source_img is None:
            print(f"警告: {filename} を読み込めませんでした。スキップします。")
            continue
        images[filename] = source_img
        loc = find_template_location(source_img, template_img)
        match_locations.append({'file': filename, 'loc': loc})
        print(f"  - {filename}: {loc}")

    if not match_locations:
        print("テンプレートを検出できる画像がありませんでした。")
        exit()

    all_x = [item['loc'][0] for item in match_locations]
    all_y = [item['loc'][1] for item in match_locations]
    target_x = int(np.median(all_x))
    target_y = int(np.median(all_y))
    print(f"\n目標座標を計算しました: ({target_x}, {target_y})")

    print("\n各画像を調整して保存しています...")
    aligned_image_filenames = []
    for item in match_locations:
        filename = item['file']
        original_loc = item['loc']
        
        dx = target_x - original_loc[0]
        dy = target_y - original_loc[1]

        print(f"  - {filename} (移動量: dx={dx}, dy={dy})")

        m = np.float32([[1, 0, dx], [0, 1, dy]])

        source_img = images[filename]
        height, width, _ = source_img.shape
        img_aligned = cv2.warpAffine(source_img, m, (width, height))

        img_cropped = crop_center(img_aligned, OUTPUT_WIDTH, OUTPUT_HEIGHT)
        output_filename = f"aligned_{filename}"
        cv2.imwrite(output_filename, img_cropped)
        aligned_image_filenames.append(output_filename)

    print("\nalign_and_crop 処理が完了しました。")

    # --- combine_images.py の処理 ---
    # aligned_IMG_*.JPG ファイルを読み込む
    # aligned_image_filenames を使用して、生成されたファイルを直接読み込む
    
    if len(aligned_image_filenames) != 3:
        print(f"エラー: 期待される3枚の 'aligned_IMG_*.JPG' ファイルが見つかりません。現在 {len(aligned_image_filenames)} 枚です。")
        exit()

    images = []
    for filename in sorted(aligned_image_filenames): # ソートして順番を保証
        img = cv2.imread(filename)
        if img is None:
            print(f"警告: {filename} を読み込めませんでした。スキップします。")
            exit()
        images.append(img)
    
    print(f"以下の3枚の画像を読み込みました: {sorted(aligned_image_filenames)}")

    height, width, _ = images[0].shape
    segment_width = width // 3

    part1 = images[0][:, width - segment_width:width]
    part2 = images[1][:, segment_width:segment_width * 2]
    part3 = images[2][:, 0:segment_width]

    combined_image = np.hstack((part3, part2, part1))

    cv2.imwrite(OUTPUT_FILENAME, combined_image)
    print(f"画像を結合し、'{OUTPUT_FILENAME}' として保存しました。")
