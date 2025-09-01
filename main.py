import cv2
import numpy as np
import glob
import os

# --- 設定 ---
OUTPUT_FILENAME = 'combined_image.jpg'
# --- 設定ここまで ---

if __name__ == '__main__':
    # 1. aligned_IMG_*.JPG ファイルを読み込む
    image_files = sorted(glob.glob('aligned_IMG_*.JPG'))

    if len(image_files) != 3:
        print(f"エラー: 期待される3枚の 'aligned_IMG_*.JPG' ファイルが見つかりません。現在 {len(image_files)} 枚です。")
        exit()

    images = []
    for filename in image_files:
        img = cv2.imread(filename)
        if img is None:
            print(f"警告: {filename} を読み込めませんでした。スキップします。")
            exit() # 読み込めないファイルがあれば処理を中断
        images.append(img)
    
    print(f"以下の3枚の画像を読み込みました: {image_files}")

    # 2. 各画像の幅と高さを取得（すべて同じサイズを想定）
    height, width, _ = images[0].shape
    segment_width = width // 3

    # 3. 各画像から必要な部分を切り出す
    # 右から古い順に並べるため、
    # 1番古い画像 (images[0]) からは右端の1/3
    # 2番目に古い画像 (images[1]) からは中央の1/3
    # 3番目に古い画像 (images[2]) からは左端の1/3
    
    # 1番古い画像 (images[0]): 右端の1/3
    # スライスは [y_start:y_end, x_start:x_end]
    part1 = images[0][:, width - segment_width:width]

    # 2番目に古い画像 (images[1]): 中央の1/3
    part2 = images[1][:, segment_width:segment_width * 2]

    # 3番目に古い画像 (images[2]): 左端の1/3
    part3 = images[2][:, 0:segment_width]

    # 4. 切り出した部分を横に結合
    # 結合順序は、右から古い順なので、part3 (左), part2 (中央), part1 (右)
    combined_image = np.hstack((part3, part2, part1))

    # 5. 結合した画像を保存
    cv2.imwrite(OUTPUT_FILENAME, combined_image)
    print(f"画像を結合し、'{OUTPUT_FILENAME}' として保存しました。")
