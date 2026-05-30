# -*- coding: utf-8 -*-
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR as RapidOCREngine

# ==================== RapidOCR 封装（从 0.py 移植） ====================
class RapidOCR:
    def __init__(self, upscale=2.0, enable_preprocess=True, **kwargs):
        """
        Args:
            upscale: ROI最大放大倍数（针对1920x1080小文字），默认2.0
            enable_preprocess: 是否启用预处理（锐化、去噪等）
            **kwargs: RapidOCR 的其他参数
        """
        self.upscale = upscale
        self.enable_preprocess = enable_preprocess

        # Default parameters
        default_kwargs = {
            'det_use_cuda': False,
            'det_limit_side_len': 1920,
            'det_limit_type': 'max',
            'det_thresh': 0.25,
            'det_box_thresh': 0.45,
            'det_unclip_ratio': 2.2,
            'det_db_score_mode': 'slow',
            'det_model_path': None,
            'rec_batch_num': 6,
            'rec_img_shape': [3, 48, 320],
            'rec_model_path': None,
            'use_angle_cls': False,
            'use_text_det': False,
            'min_height': 25,
            'width_height_ratio': 8,
            'text_score': 0.5,
            'print_verbose': False,
            'max_side_len': 2000,
            'min_side_len': 30,
        }

        final_kwargs = default_kwargs.copy()
        final_kwargs.update(kwargs)
        self.engine = RapidOCREngine(**final_kwargs)

    def _preprocess_roi(self, roi):
        """预处理ROI区域（放大、锐化、去噪）"""
        h, w = roi.shape[:2]

        # 1. 放大小图
        target_min_h = 48
        if h > 0 and target_min_h and self.upscale and self.upscale > 1:
            scale = target_min_h / float(h)
            scale = max(1.0, min(float(self.upscale), scale))
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if not self.enable_preprocess:
            return roi

        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # 2. 去噪
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            try:
                roi = cv2.fastNlMeansDenoisingColored(roi, None, 7, 7, 7, 21)
            except Exception:
                pass

        # 3. 锐化 + CLAHE
        try:
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            pass

        blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=1.0)
        roi = cv2.addWeighted(roi, 1.6, blurred, -0.6, 0)
        return roi

    def recognize_single_roi(self, img, box, preprocess=True, return_raw=False):
        if isinstance(img, str):
            img = cv2.imread(img)
        x, y, w, h = box
        roi = img[y:y + h, x:x + w].copy()
        if preprocess:
            roi = self._preprocess_roi(roi)
        result = self.engine(roi)

        # 兼容 rapidocr 不同版本的返回值
        if hasattr(result, 'txts') and result.txts:
            text = result.txts[0]
            score = result.scores[0] if hasattr(result, 'scores') else 0.0
            raw_text = text
        elif hasattr(result, 'rec_texts') and result.rec_texts:
            text = result.rec_texts[0]
            score = result.rec_scores[0] if result.rec_scores else 0.0
            raw_text = text
        elif isinstance(result, tuple) and len(result) == 2:
            res_data, _ = result
            if res_data and len(res_data) > 0:
                text = res_data[0][1]
                score = float(res_data[0][2])
                raw_text = text
            else:
                text = score = None
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            if isinstance(result[0], (list, tuple)) and len(result[0]) > 1:
                text = result[0][1]
                score = float(result[0][2]) if len(result[0]) > 2 else 0.0
            else:
                text = None
                score = 0.0
            raw_text = text
        else:
            text = score = None

        if text is None:
            if return_raw:
                return None, 0.0, None
            return None, 0.0

        if return_raw:
            return text, score, raw_text
        return text, score


# ==================== 全局 OCR 引擎（只初始化一次） ====================
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR(upscale=3.0, enable_preprocess=True)
    return _ocr_engine


# ==================== 球种识别函数（使用 RapidOCR） ====================
def ocr_current_ball(gray):
    """
    使用 RapidOCR 识别当前球种
    :param gray: 灰度图（numpy数组），尺寸应与原图一致（例如960x540或1920x1080）
    :return: 球种名称字符串
    """
    # 将灰度图转为 BGR（RapidOCR 需要彩色图）
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 裁剪区域（与之前 Tesseract 版本保持一致）
    crop_x, crop_y, crop_w, crop_h = 726, 336, 68, 30

    ocr = get_ocr_engine()
    text, score = ocr.recognize_single_roi(bgr, (crop_x, crop_y, crop_w, crop_h))

    if text is None:
        return ""

    # 清理文本（去除空白）
    text = "".join(text.split())

    # 后处理修正（兼容 OCR 可能的误识别）
    '''
    if "精" in text or "灵" in text:
        text = "精灵球"
    elif text == "治球":
    '''
    if text == "治球":
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
    """
    识别当前球种的剩余数量（保持原有 Tesseract 逻辑，也可改为 RapidOCR，但数量识别已经很稳定）
    """
    crop_x, crop_y, crop_w, crop_h = 879, 342, 37, 21
    crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    crop_gray = cv2.resize(crop_gray, (crop_w*5, crop_h*5))
    _, thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    closing = cv2.resize(closing, (crop_w, crop_h))

    # 数量识别使用 Tesseract 更简单
    import pytesseract
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

    # 保存裁剪区域供检查
    crop_ball = img[336:336+30, 726:726+68]
    cv2.imwrite("debug_ball_crop.png", crop_ball)
    print("球种裁剪区域已保存为 debug_ball_crop.png")
    crop_num = img[342:342+21, 879:879+37]
    cv2.imwrite("debug_num_crop.png", crop_num)
    print("数量裁剪区域已保存为 debug_num_crop.png")


if __name__ == "__main__":
    main()