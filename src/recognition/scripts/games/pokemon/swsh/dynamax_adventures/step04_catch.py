from enum import Enum
from recognition.scripts.base.base_script import BaseScript
from recognition.scripts.base.base_sub_step import BaseSubStep, SubStepRunningStatus
import time
import cv2
import numpy as np
import pytesseract
from recognition.ocr.rapidocr import RapidOCR
import os
from datetime import datetime

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
                if self._target_ball == '究极球' or self._target_ball == '诱饵球' or self._target_ball == "速度球" or self._target_ball == "梦境球":
                    self.script.macro_text_run("RIGHT:0.1", block=True)
                else:
                    self.script.macro_text_run("LEFT:0.1", block=True)
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