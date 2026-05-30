# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pytesseract

# 如果 Tesseract 不在 PATH 中，请取消下一行的注释并修改为你的实际路径
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_current_ball(gray):
    crop_x, crop_y, crop_w, crop_h = 726, 336, 68, 30
    crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    # 放大5倍
    crop_gray = cv2.resize(crop_gray, (crop_w*5, crop_h*5))
    # 二值化：黑字白底（不反转）
    _, thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 使用稍大的核连接文字（核大小3x3）
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 闭运算填充空洞
    
    # 识别（暂时不加白名单，先看原始输出）
    text = pytesseract.image_to_string(thresh, lang='chi_sim')
    text = "".join(text.split())
    
    # 后处理修正
    if "精" in text or "灵" in text:
        text = "精灵球"
    elif text == "治球":
        text = "治愈球"
    elif text == "愈球":
        text = "治愈球"
    elif text == "潜球":
        text = "巢穴球"
    elif text == "球":
        text = "巢穴球"
    elif text == "暗球":
        text = "黑暗球"
    elif text == "华球":
        text = "豪华球"
    elif text == "重球":
        text = "沉重球"
    elif text == "境球":
        text = "梦境球"
    elif text == "饵球":
        text = "诱饵球"
    return text

def ocr_current_ball_left_num(gray):
    crop_x, crop_y, crop_w, crop_h = 879, 342, 37, 21
    crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    crop_gray = cv2.resize(crop_gray, (crop_w*5, crop_h*5))
    # 改为 cv2.THRESH_BINARY（黑字白底），不反转
    _, thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 减小核大小，避免数字粘连
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   # 开运算去噪点
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) # 闭运算填补小洞
    closing = cv2.resize(closing, (crop_w, crop_h))
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(closing, config=custom_config)
    text = "".join(text.split())
    num = int(text) if text.isdigit() else 0
    return num

def main():
    img = cv2.imread("debug_catch.png")
    if img is None:
        print("错误：无法读取 debug_catch.png，请确保文件存在。")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ball = ocr_current_ball(gray)
    num = ocr_current_ball_left_num(gray)
    print(f"识别结果：球种 = '{ball}'，剩余数量 = {num}")
    
    # 保存裁剪区域（使用与识别函数一致的坐标）
    crop_ball = img[336:336+30, 726:726+68]   # 对应 ocr_current_ball 的裁剪区域
    cv2.imwrite("debug_ball_crop.png", crop_ball)
    print("球种裁剪区域已保存为 debug_ball_crop.png")
    crop_num = img[342:342+21, 879:879+37]    # 对应 ocr_current_ball_left_num 的裁剪区域
    cv2.imwrite("debug_num_crop.png", crop_num)
    print("数量裁剪区域已保存为 debug_num_crop.png")

if __name__ == "__main__":
    main()