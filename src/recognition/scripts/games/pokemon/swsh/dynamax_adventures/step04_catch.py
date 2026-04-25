from enum import Enum
from recognition.scripts.base.base_script import BaseScript
from recognition.scripts.base.base_sub_step import BaseSubStep, SubStepRunningStatus
import time
import cv2
import numpy as np
import pytesseract
from rapidocr_onnxruntime import RapidOCR as RapidOCREngine
import os
from datetime import datetime


# ---------- 移植 RapidOCR 封装类 ----------
class RapidOCR:
    def __init__(self, upscale=2.0, enable_preprocess=True, **kwargs):
        self.upscale = upscale
        self.enable_preprocess = enable_preprocess
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
        h, w = roi.shape[:2]
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
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            try:
                roi = cv2.fastNlMeansDenoisingColored(roi, None, 7, 7, 7, 21)
            except Exception:
                pass
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
        roi = img[y:y+h, x:x+w].copy()
        if preprocess:
            roi = self._preprocess_roi(roi)
        result = self.engine(roi)
        if hasattr(result, 'txts') and result.txts:
            text = result.txts[0]
            score = result.scores[0] if hasattr(result, 'scores') else 0.0
        elif hasattr(result, 'rec_texts') and result.rec_texts:
            text = result.rec_texts[0]
            score = result.rec_scores[0] if result.rec_scores else 0.0
        elif isinstance(result, tuple) and len(result) == 2:
            res_data, _ = result
            if res_data and len(res_data) > 0:
                text = res_data[0][1]
                score = float(res_data[0][2])
            else:
                text = None
                score = 0.0
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            if isinstance(result[0], (list, tuple)) and len(result[0]) > 1:
                text = result[0][1]
                score = float(result[0][2]) if len(result[0]) > 2 else 0.0
            else:
                text = None
                score = 0.0
        else:
            text = None
            score = 0.0
        if return_raw:
            return text, score, text
        return text, score


class SWSHDACatchResult(Enum):
    NotCaught = 0
    Caught = 1


class SWSHDACatch(BaseSubStep):
    _rapid_ocr = None

    @classmethod
    def _get_ocr(cls):
        if cls._rapid_ocr is None:
            cls._rapid_ocr = RapidOCR(upscale=3.0, enable_preprocess=True)
        return cls._rapid_ocr

    def __init__(self, script: BaseScript, battle_index: int = 0, catch: bool = True, target_ball: str = "究极球", timeout: float = 60) -> None:
        super().__init__(script, timeout)
        self._process_step_index = 0
        self._target_ball = target_ball
        self._catch_flag = catch
        if battle_index >= 3:
            self._catch_flag = True
        self._catch_result = SWSHDACatchResult.NotCaught
        self._fail_saved = False

    @property
    def catch_result(self):
        return self._catch_result

    def _process(self) -> SubStepRunningStatus:
        self._status = self.running_status
        if self._process_step_index >= 0:
            if self._process_step_index >= len(self._process_steps):
                return SubStepRunningStatus.OK
            elif self._status == SubStepRunningStatus.Running:
                self._process_steps[self._process_step_index]()
                return self._status
            else:
                return self._status
        else:
            self._process_step_index = 0
            return self._process()

    @property
    def _process_steps(self):
        return [
            self._process_step_0,
            self._process_step_1,
        ]

    def _process_step_0(self):
        if self._catch_flag:
            self.script.macro_text_run("A:0.1", block=True)
            self.time_sleep(0.5)
            self._last_action_time_monotonic = time.monotonic()
            self._initial_ball = None
            self._process_step_index += 1
            self._fail_saved = False
        else:
            self.script.macro_text_run("BOTTOM:0.1->0.5->A:0.1", block=True)
            self.time_sleep(0.5)
            self._status = SubStepRunningStatus.OK

    def _process_step_1(self):
        current_frame_960x540 = self.script.current_frame_960x540
        gray_frame = cv2.cvtColor(current_frame_960x540, cv2.COLOR_BGR2GRAY)

        current_ball = self._ocr_current_ball(gray_frame)
        num = self._ocr_current_ball_left_num(gray_frame)

        #self.script.send_log(f"[捕捉] 识别结果: 球种='{current_ball}', 剩余数量={num}")

        if current_ball == "" and not self._fail_saved:
            self._save_debug_screenshot(current_frame_960x540, gray_frame)
            self._fail_saved = True

        if current_ball == "":
            return
        else:
            if self._initial_ball is None:
                self._initial_ball = current_ball
            elif current_ball == self._initial_ball:
                self._target_ball = None
                self._initial_ball = "不检测"

            catch_flag = False
            if self._target_ball is None:
                catch_flag = (self._alternatives(current_ball) and num > 0)
            else:
                if current_ball == self._target_ball and num > 0:
                    catch_flag = True
            if catch_flag:
                self.script.macro_text_run("A:0.1", block=True)
                self.time_sleep(8)
                self._last_action_time_monotonic = time.monotonic()
                self._catch_result = SWSHDACatchResult.Caught

                
                self.script._total_catch_count += 1
                

                self._process_step_index += 1
            else:
                if self._target_ball == '究极球':
                    self.script.macro_text_run("LEFT:0.1", block=True)
                else:
                    self.script.macro_text_run("RIGHT:0.1", block=True)
                self.time_sleep(0.5)
                self._last_action_time_monotonic = time.monotonic()

    def _ocr_current_ball(self, gray):
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        crop_x, crop_y, crop_w, crop_h = 726, 336, 68, 30
        ocr = self._get_ocr()
        text_raw, score, _ = ocr.recognize_single_roi(bgr, (crop_x, crop_y, crop_w, crop_h), preprocess=True, return_raw=True)
        if text_raw is None:
            self.script.send_log(f"[OCR] 球种识别失败: 未返回文本")
            return ""
        #self.script.send_log(f"[OCR] 原始识别: '{text_raw}' (置信度: {score:.4f})")
        text = "".join(text_raw.split())
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

    def _ocr_current_ball_left_num(self, gray):
        crop_x, crop_y, crop_w, crop_h = 879, 342, 37, 21
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        crop_gray = cv2.resize(crop_gray, (crop_w*5, crop_h*5))
        _, thresh1 = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        kernel = np.ones((6, 6), np.uint8)
        # MODIFIED: 修复形态学操作参数，将 cv2.THRESH_OTSU 改为 cv2.MORPH_OPEN
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(closing, config=custom_config)
        text = "".join(text.split())
        num = int(text) if text.isdigit() else 0
        return num

    def _alternatives(self, ball):
        if ball == "精灵球":
            return True
        elif ball == "豪华球":
            return True
        else:
            return False

    def _save_debug_screenshot(self, frame_bgr, gray_frame):
        try:
            save_dir = "ocr_fail_screenshots"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            full_path = os.path.join(save_dir, f"catch_fail_full_{timestamp}.png")
            cv2.imwrite(full_path, frame_bgr)
            self.script.send_log(f"[调试] 已保存完整截图: {full_path}")
            crop_x, crop_y, crop_w, crop_h = 726, 336, 68, 30
            crop_ball = frame_bgr[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            ball_path = os.path.join(save_dir, f"catch_fail_ball_{timestamp}.png")
            cv2.imwrite(ball_path, crop_ball)
            self.script.send_log(f"[调试] 已保存球种裁剪区域: {ball_path}")
            num_x, num_y, num_w, num_h = 879, 342, 37, 21
            crop_num = frame_bgr[num_y:num_y+num_h, num_x:num_x+num_w]
            num_path = os.path.join(save_dir, f"catch_fail_num_{timestamp}.png")
            cv2.imwrite(num_path, crop_num)
            self.script.send_log(f"[调试] 已保存数量裁剪区域: {num_path}")
        except Exception as e:
            self.script.send_log(f"[调试] 保存截图失败: {e}")