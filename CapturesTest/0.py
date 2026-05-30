import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR as RapidOCREngine

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
            # === 检测模块参数 ===
            'det_use_cuda': False,
            'det_limit_side_len': 1920,
            'det_limit_type': 'max',
            'det_thresh': 0.25,
            'det_box_thresh': 0.45,
            'det_unclip_ratio': 2.2,
            'det_db_score_mode': 'slow',
            'det_model_path': None,

            # === 识别模块参数 ===
            'rec_batch_num': 6,
            'rec_img_shape': [3, 48, 320],
            'rec_model_path': None,

            # === 全局参数 ===
            'use_angle_cls': False,
            'use_text_det': False,
            'min_height': 25,
            'width_height_ratio': 8,
            'text_score': 0.5,
            'print_verbose': False,

            # === 图像尺寸限制 ===
            'max_side_len': 2000,
            'min_side_len': 30,
        }

        # Update defaults with provided kwargs
        final_kwargs = default_kwargs.copy()
        final_kwargs.update(kwargs)

        # 初始化引擎
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

        # 2. 去噪（可选）
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            try:
                roi = cv2.fastNlMeansDenoisingColored(roi, None, 7, 7, 7, 21)
            except Exception:
                pass

        # 3. 锐化
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


    def batch_recognize_regions(self, img, regions, return_details=False):
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise ValueError(f"无法读取图像: {img}")

        is_dict = isinstance(regions, dict)
        if is_dict:
            region_items = list(regions.items())
        else:
            region_items = [(f"roi_{i}", box) for i, box in enumerate(regions)]

        results = {} if is_dict else []

        for name, box in region_items:
            x, y, w, h = box
            try:
                roi = img[y:y+h, x:x+w].copy()
            except Exception as e:
                result_item = {'text': None, 'score': 0.0, 'error': f'裁剪失败: {str(e)}'}
                if return_details:
                    result_item['box'] = (x, y, w, h)
                if is_dict:
                    results[name] = result_item
                else:
                    results.append(result_item)
                continue

            roi_processed = self._preprocess_roi(roi)
            try:
                result, elapse = self.engine(roi_processed)  # 注意：新版库可能不返回elapse，可能需要调整
                # 兼容新版 rapidocr (v3.x) 返回格式: result = self.engine(roi_processed)
                # 如果报错，可以改为 result = self.engine(roi_processed)
                if result and len(result) > 0:
                    # 根据不同版本，result 可能是列表或元组
                    if isinstance(result, (list, tuple)) and len(result) > 0:
                        # 常见格式: [[[[x1,y1,x2,y2,...], '文本', score], ...], ...]
                        # 这里简化处理，假设 result[0][1] 是文本，result[0][2] 是分数
                        text = result[0][1]
                        score = float(result[0][2])
                    else:
                        text = None
                        score = 0.0
                    result_item = {'text': text, 'score': score}
                    if return_details:
                        result_item['box'] = (x, y, w, h)
                        result_item['elapse'] = elapse if 'elapse' in locals() else 0
                else:
                    result_item = {'text': None, 'score': 0.0}
                    if return_details:
                        result_item['box'] = (x, y, w, h)
            except Exception as e:
                result_item = {'text': None, 'score': 0.0, 'error': f'识别失败: {str(e)}'}
                if return_details:
                    result_item['box'] = (x, y, w, h)

            if is_dict:
                results[name] = result_item
            else:
                results.append(result_item)
        return results

    def recognize_single_roi(self, img, box, preprocess=True, return_raw=False):
        if isinstance(img, str):
            img = cv2.imread(img)
        x, y, w, h = box
        roi = img[y:y + h, x:x + w].copy()
        if preprocess:
            roi = self._preprocess_roi(roi)
        result = self.engine(roi)

        # 新版 rapidocr (v3.x) 返回对象属性为 txts 和 scores
        if hasattr(result, 'txts') and result.txts:
            text = result.txts[0]
            score = result.scores[0] if hasattr(result, 'scores') else 0.0
            raw_text = text
        # 兼容旧版（如有 rec_texts）
        elif hasattr(result, 'rec_texts') and result.rec_texts:
            text = result.rec_texts[0]
            score = result.rec_scores[0] if result.rec_scores else 0.0
            raw_text = text
        # 兼容旧版返回元组 (result_data, elapse)
        elif isinstance(result, tuple) and len(result) == 2:
            res_data, _ = result
            if res_data and len(res_data) > 0:
                text = res_data[0][1]
                score = float(res_data[0][2])
                raw_text = text
            else:
                text = score = None
        # 兼容旧版直接返回列表
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


if __name__ == "__main__":
    ocr = RapidOCR(upscale=3.0, enable_preprocess=True)
    img_path = r"D:\software\pokemon-controller-gui-main\src\Captures\20260513182316.jpg"

    regions = [
        (693*2,(353+52*3)*2,50*2,16*2)
    ]
    for region in regions:
        text, score = ocr.recognize_single_roi(img_path, region)
        print(f"识别结果: {text}")
        print(f"置信度: {score:.4f}")
    import cv2

    img = cv2.imread(img_path)

    for (x, y, w, h) in regions:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("regions", img)
    cv2.waitKey(0)