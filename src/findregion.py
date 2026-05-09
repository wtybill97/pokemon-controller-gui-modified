# -*- coding: utf-8 -*-
"""
定位图片中指定文字区域的工具
依赖: easyocr, opencv-python, matplotlib
安装: pip install easyocr opencv-python matplotlib
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from recognition.ocr.easy import EasyOCR   # 导入你提供的封装类

def find_text_region(image_path, target_text, use_angle_cls=True, show_result=True):
    """
    在图片中查找指定文字所在的区域（基于 EasyOCR）

    Args:
        image_path (str): 图片路径
        target_text (str): 要查找的文字（如 "原来如此，原来如此。"）
        use_angle_cls (bool): 是否使用角度分类（EasyOCR 中无此参数，保留仅为兼容）
        show_result (bool): 是否显示结果图片

    Returns:
        tuple: (box, cropped_img) 
               box - 文字区域的四个顶点坐标，格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
               cropped_img - 裁剪后的图像区域（如果找到），否则为 None
    """
    # 获取 EasyOCR 单例（需要中文支持）
    # EasyOCR 类的 __new__ 会维护单例，这里直接调用构造函数获取实例
    # 注意：默认 langs 为 'en'，需要改为 'ch_sim,en' 以支持简体中文
    # 为了避免重新初始化，我们先检查已有实例的语种。简单处理：直接创建新实例（单例会复用）
    ocr_instance = EasyOCR(langs='ch_sim,en', gpu=True, verbose=False)
    reader = ocr_instance._reader   # 获取内部的 easyocr.Reader 对象

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 执行 EasyOCR 识别，返回带边界框的结果
    # readtext 返回列表: [ (bbox, text, confidence), ... ]
    # bbox 格式: [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
    result = reader.readtext(image_path, paragraph=False, detail=1)

    if not result:
        print("未检测到任何文字")
        return None, None

    # 遍历所有检测到的文本框
    target_box = None
    target_text_found = None
    for bbox, text, conf in result:
        if target_text in text:   # 支持部分匹配
            target_box = bbox
            target_text_found = text
            print(f"找到目标文字: '{text}' (置信度: {conf:.2f})")
            break

    if target_box is None:
        print(f"未找到文字 '{target_text}'")
        return None, None

    # 将四个顶点转换为整数坐标
    box = np.array(target_box, dtype=np.int32)

    # 计算最小外接矩形（用于裁剪）
    x_min = min(box[:, 0])
    x_max = max(box[:, 0])
    y_min = min(box[:, 1])
    y_max = max(box[:, 1])
    cropped = img_rgb[y_min:y_max, x_min:x_max]

    # 可选：在原图上绘制边界框并显示
    if show_result:
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
    # 使用原始字符串避免转义警告
    image_file = r"D:\software\pokemon-controller-gui-main-20260414\Captures\20260506140436.jpg"
    target = "漂亮地捉住"

    box, cropped_img = find_text_region(image_file, target, show_result=True)

    if box is not None:
        print("\n文字区域四点坐标:")
        print(box.tolist())
        x_vals = box[:, 0]
        y_vals = box[:, 1]
        print(f"外接矩形: x={min(x_vals)}, y={min(y_vals)}, w={max(x_vals)-min(x_vals)}, h={max(y_vals)-min(y_vals)}")

        if cropped_img is not None:
            cv2.imwrite("cropped_text_region.jpg", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
            print("裁剪后的文字区域已保存为 cropped_text_region.jpg")

if __name__ == "__main__":
    main()