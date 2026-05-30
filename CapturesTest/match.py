# test_arrow_match.py
import cv2
import numpy as np
import os

# ========== 在这里直接设置路径 ==========
IMAGE_PATH = r"D:\software\pokemon-controller-gui-main\src\Captures\20260511142631.jpg"  # 替换为你的截图路径
TEMPLATE_PATH = r"D:\software\pokemon-controller-gui-main-20260414\entrance.jpg"
THRESHOLD = 0.7  # 匹配阈值，与原 battle.py 一致


# =================================

def match_arrow(image_path, template_path, threshold=0.7):
    """
    验证图片在指定区域(620,320,46,220)是否能匹配到箭头模板

    Returns:
        (is_matched, max_val, match_location_in_roi)
    """
    # 1. 读取并缩放至 960x540
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    resized = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 2. 加载模板
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"无法读取模板: {template_path}")

    # 3. 裁剪 ROI (x=620, y=320, w=46, h=220)
    x, y, w, h = 792, 318, 48, 28
    roi = gray[y:y + h, x:x + w]

    # 4. 模板匹配
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    is_matched = max_val >= threshold
    return is_matched, max_val, max_loc


def visualize_match(image_path, template_path, output_path=None, threshold=0.7):
    """可视化匹配结果，可选保存或显示"""
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    resized = cv2.resize(img, (960, 540))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("无法读取模板")
        return

    x, y, w, h = 792, 318, 48, 28
    roi = gray[y:y + h, x:x + w]
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    match_x = x + max_loc[0]
    match_y = y + max_loc[1]
    tw, th = template.shape[1], template.shape[0]
    cv2.rectangle(resized, (match_x, match_y), (match_x + tw, match_y + th), (0, 255, 0), 2)
    cv2.putText(resized, f"score: {max_val:.3f}", (match_x, match_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if output_path:
        cv2.imwrite(output_path, resized)
        print(f"结果图片已保存至: {output_path}")
    else:
        cv2.imshow("Arrow Match Result", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 检查路径有效性
    if not os.path.exists(IMAGE_PATH):
        print(f"错误: 图片文件不存在: {IMAGE_PATH}")
        exit(1)
    if not os.path.exists(TEMPLATE_PATH):
        print(f"错误: 模板文件不存在: {TEMPLATE_PATH}")
        exit(1)

    # 执行匹配
    matched, score, loc = match_arrow(IMAGE_PATH, TEMPLATE_PATH, THRESHOLD)
    print(f"匹配结果: {'成功' if matched else '失败'}")
    print(f"最大相似度: {score:.4f}")
    if matched:
        print(f"匹配位置(裁剪区域内): x={loc[0]}, y={loc[1]}")

    # 询问是否显示可视化
    print("\n是否显示匹配结果可视化？(y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        visualize_match(IMAGE_PATH, TEMPLATE_PATH, threshold=THRESHOLD)