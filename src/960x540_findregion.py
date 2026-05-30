# -*- coding: utf-8 -*-
"""
定位图片中指定文字区域的工具（支持先缩放至960x540）
依赖: easyocr, opencv-python, matplotlib
安装: pip install easyocr opencv-python matplotlib
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
from matplotlib import pyplot as plt
from recognition.ocr.easy import EasyOCR   # 导入你提供的封装类

def find_text_region(image_path, target_text, use_angle_cls=True, show_result=True):
    """
    在图片中查找指定文字所在的区域（先将图片缩放至960x540，再识别）

    Args:
        image_path (str): 图片路径
        target_text (str): 要查找的文字（如 "漂亮地捉住"）
        use_angle_cls (bool): 保留参数，兼容性
        show_result (bool): 是否显示结果图片（显示缩放后的图片与检测结果）

    Returns:
        tuple: (box, cropped_img_resized)
               box - 文字区域在缩放后图像(960x540)上的四个顶点坐标，格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
               cropped_img_resized - 从缩放后图像中裁剪出的文字区域（RGB格式）
    """
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    ocr_instance = EasyOCR(langs='ch_sim,en', gpu=True, verbose=False)
    # 1. 获取 EasyOCR 实例（支持简体中文）
    ocr_instance = EasyOCR(langs='ch_sim,en', gpu=True, verbose=False)
    reader = ocr_instance._reader   # 内部的 easyocr.Reader

    # 2. 读取原始图像
    img_original = cv2.imread(image_path)
    if img_original is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    # 3. 缩放至 960x540 (参照 BaseScript.current_frame_960x540 中的方法)
    target_width, target_height = 960, 540
    img_resized = cv2.resize(img_original, (target_width, target_height), interpolation=cv2.INTER_AREA)
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 4. 在缩放后的图像上执行 OCR
    # readtext 可以直接接受 numpy 数组 (RGB 或 BGR)
    result = reader.readtext(img_resized_rgb, paragraph=False, detail=1)

    if not result:
        print("未检测到任何文字")
        return None, None

    # 5. 查找目标文字
    target_box = None
    target_text_found = None
    for bbox, text, conf in result:
        if target_text in text:
            target_box = bbox
            target_text_found = text
            print(f"找到目标文字: '{text}' (置信度: {conf:.2f})")
            break

    if target_box is None:
        print(f"未找到文字 '{target_text}'")
        return None, None

    # 6. 转换坐标为整数
    box = np.array(target_box, dtype=np.int32)

    # 7. 计算缩放图上的外接矩形并裁剪
    x_min = min(box[:, 0])
    x_max = max(box[:, 0])
    y_min = min(box[:, 1])
    y_max = max(box[:, 1])
    cropped_resized = img_resized_rgb[y_min:y_max, x_min:x_max]

    # 8. 显示结果（可选）
    if show_result:
        img_copy = img_resized_rgb.copy()
        cv2.polylines(img_copy, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img_resized_rgb)
        plt.title("缩放后图片 (960x540)")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img_copy)
        plt.title("检测结果（绿色框为目标区域）")
        plt.axis('off')
        plt.show()

    return box, cropped_resized


def main():
    image_file = r"D:\software\pokemon-controller-gui-main-20260414\Captures\20260530175240.jpg"
    target = "帕路奇亚"

    box, cropped_img = find_text_region(image_file, target, show_result=True)

    if box is not None:
        print("\n文字区域在缩放后图像(960x540)上的四点坐标:")
        print(box.tolist())
        x_vals = box[:, 0]
        y_vals = box[:, 1]
        print(f"外接矩形(缩放图): x={min(x_vals)}, y={min(y_vals)}, w={max(x_vals)-min(x_vals)}, h={max(y_vals)-min(y_vals)}")

        if cropped_img is not None:
            # 保存缩放图上的裁剪区域（如果需要在原图上裁剪，可相应变换坐标）
            cv2.imwrite("cropped_text_region_resized.jpg", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
            print("裁剪后的文字区域（缩放图）已保存为 cropped_text_region_resized.jpg")

if __name__ == "__main__":
    main()