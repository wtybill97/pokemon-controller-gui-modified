import cv2
import numpy as np
import easyocr
from matplotlib import pyplot as plt

def find_text_region_easyocr(image_path, target_text):
    """
    使用 EasyOCR 定位图片中的指定文字区域
    返回: (box, cropped_img)
        box: 四点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        cropped_img: 裁剪后的 RGB 图像
    """
    # 初始化 EasyOCR（会自动下载中英文模型）
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)  # gpu=True 需要 CUDA

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 执行 OCR，返回 (bbox, text, confidence)
    results = reader.readtext(img_rgb)

    target_box = None
    target_text_found = None
    for (bbox, text, conf) in results:
        if target_text in text:
            target_box = bbox
            target_text_found = text
            print(f"✅ 找到目标文字: '{text}' (置信度: {conf:.2f})")
            break

    if target_box is None:
        print(f"❌ 未找到文字 '{target_text}'")
        return None, None

    # 转换为整数坐标
    box = np.array(target_box, dtype=np.int32)
    x_min = min(box[:, 0])
    x_max = max(box[:, 0])
    y_min = min(box[:, 1])
    y_max = max(box[:, 1])
    cropped = img_rgb[y_min:y_max, x_min:x_max]

    # 可视化
    img_copy = img_rgb.copy()
    cv2.polylines(img_copy, [box], isClosed=True, color=(0, 255, 0), thickness=2)

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("原始图片")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_copy)
    plt.title("检测结果（绿色框为目标区域）")
    plt.axis('off')
    plt.show()

    return box, cropped

def main():
    # 注意：Windows路径推荐使用原始字符串 r"..."
    image_file = r"D:\software\pokemon-controller-gui-main-20260414\Captures\20260503131543.jpg"
    target = "你替我捉住"

    box, cropped = find_text_region_easyocr(image_file, target)

    if box is not None:
        print("\n📐 文字区域四点坐标:")
        print(box.tolist())
        x_vals = box[:, 0]
        y_vals = box[:, 1]
        print(f"外接矩形: x={min(x_vals)}, y={min(y_vals)}, w={max(x_vals)-min(x_vals)}, h={max(y_vals)-min(y_vals)}")

        if cropped is not None:
            output_path = "cropped_text_region.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            print(f"💾 裁剪区域已保存为 {output_path}")

if __name__ == "__main__":
    main()