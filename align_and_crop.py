

import cv2
import numpy as np
import glob
import os

# --- 設定 ---
# 出力画像のサイズ
OUTPUT_WIDTH = 5700
OUTPUT_HEIGHT = 3900
TEMPLATE_FILENAME = 'target.jpg'
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
    # --- 1. テンプレートとソース画像を読み込む ---
    try:
        template_img = cv2.imread(TEMPLATE_FILENAME)
        if template_img is None:
            raise FileNotFoundError
        print(f"テンプレート画像を読み込みました: {TEMPLATE_FILENAME}")
    except (FileNotFoundError, cv2.error):
        print(f"エラー: テンプレート画像 '{TEMPLATE_FILENAME}' が見つからないか、読み込めません。")
        exit()

    image_files = sorted(glob.glob('IMG_*.JPG'))
    if not image_files:
        print("処理対象のIMG_*.JPGファイルが見つかりません。")
        exit()

    # --- 2. 各画像内のテンプレートの位置を検出 ---
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

    # --- 3. 目標座標を計算（中央値） ---
    all_x = [item['loc'][0] for item in match_locations]
    all_y = [item['loc'][1] for item in match_locations]
    target_x = int(np.median(all_x))
    target_y = int(np.median(all_y))
    print(f"\n目標座標を計算しました: ({target_x}, {target_y})")

    # --- 4. 各画像を調整して保存 ---
    print("\n各画像を調整して保存しています...")
    for item in match_locations:
        filename = item['file']
        original_loc = item['loc']
        
        # 目標座標に合わせるための移動量を計算
        dx = target_x - original_loc[0]
        dy = target_y - original_loc[1]

        print(f"  - {filename} (移動量: dx={dx}, dy={dy})")

        # アフィン変換行列を作成
        m = np.float32([[1, 0, dx], [0, 1, dy]])

        # アフィン変換を適用
        source_img = images[filename]
        height, width, _ = source_img.shape
        img_aligned = cv2.warpAffine(source_img, m, (width, height))

        # 中央をクロップして保存
        img_cropped = crop_center(img_aligned, OUTPUT_WIDTH, OUTPUT_HEIGHT)
        output_filename = f"aligned_{filename}"
        cv2.imwrite(output_filename, img_cropped)

    print("\nすべての処理が完了しました。")
