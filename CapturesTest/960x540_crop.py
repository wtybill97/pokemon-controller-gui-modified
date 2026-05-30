import cv2

def resize_and_crop(image_path, rect_on_resized, save_path):
    """
    先将原始图像缩放到 960x540，然后从缩放图上裁剪指定区域
    :param image_path:     原始图片路径
    :param rect_on_resized: 缩放图上的矩形 (x, y, w, h)，例如 (792, 318, 48, 28)
    :param save_path:      保存裁剪结果的文件路径
    """
    # 1. 读取原始图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return

    # 2. 缩放至 960x540
    target_w, target_h = 960, 540
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 3. 从缩放图上裁剪（直接使用给出的坐标）
    x, y, w, h = rect_on_resized

    # 边界修正（防止超出图像范围）
    x = max(0, min(x, target_w - 1))
    y = max(0, min(y, target_h - 1))
    w = min(w, target_w - x)
    h = min(h, target_h - y)

    if w <= 0 or h <= 0:
        print("裁剪区域无效")
        return

    cropped = resized[y:y+h, x:x+w]

    # 4. 保存
    cv2.imwrite(save_path, cropped)
    print(f"裁剪区域已保存至: {save_path}")
    print(f"实际裁剪坐标(缩放图): x={x}, y={y}, w={w}, h={h}")
    print(f"裁剪图像尺寸: {cropped.shape[1]} x {cropped.shape[0]}")

# 使用示例
if __name__ == "__main__":
    image_path = r"D:\software\pokemon-controller-gui-main\src\Captures\20260511142631.jpg"
    # 注意：这里的坐标是缩放图(960x540)上的坐标
    rect_on_resized = (440, 165, 48, 28)
    save_path = "cropped_region.jpg"

    resize_and_crop(image_path, rect_on_resized, save_path)