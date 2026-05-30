from enum import Enum
from recognition.scripts.base.base_script import BaseScript
from recognition.scripts.base.base_sub_step import BaseSubStep, SubStepRunningStatus
import cv2
import re
from recognition.ocr.rapidocr import RapidOCR
from recognition.scripts.games.pokemon.swsh.common.image_match.checkbox_match import ChatBoxMatch
from recognition.scripts.games.pokemon.swsh.common.image_match.pokemon_detail_shiny_match import PokemonDetailShinyMatch

# 复用通用 OCR 引擎（与 battle 中一致）
_shared_ocr = None
def get_ocr():
    global _shared_ocr
    if _shared_ocr is None:
        _shared_ocr = RapidOCR(upscale=3.0, enable_preprocess=True)
    return _shared_ocr


class SWSHDAFinish(BaseSubStep):
    def __init__(self, script: BaseScript, timeout: float = 30) -> None:
        super().__init__(script, timeout)
        self._process_step_index = 0
        self._check_counter = 0
        self._get_rewards_label_template = cv2.imread(
            "resources/img/recognition/pokemon/swsh/dynamax_adventures/get_rewards_label.png")
        self._get_rewards_label_template = cv2.cvtColor(
            self._get_rewards_label_template, cv2.COLOR_BGR2GRAY)
        self._chatbox_template = cv2.imread(
            "resources/img/recognition/pokemon/swsh/dynamax_adventures/chatbox.png")
        self._chatbox_template = cv2.cvtColor(
            self._chatbox_template, cv2.COLOR_BGR2GRAY)

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
            self._process_steps_0,
            self._process_steps_1,
            self._process_steps_2,
        ]

    def _process_steps_0(self):
        current_frame_960x540 = self.script.current_frame_960x540
        gray_frame = cv2.cvtColor(current_frame_960x540, cv2.COLOR_BGR2GRAY)
        if not self._match_get_rewards_page(gray_frame):
            self.time_sleep(0.5)
            return
        self._process_step_index += 1

    def _process_steps_1(self):
        # 原有按键序列
        self.script.macro_text_run("A:0.1->3->B:0.1->0.5->B:0.1->0.5->A:0.1", block=True)
        self.time_sleep(0.5)
        
        # ========== 识别极矿石数量（仅第一次进入时识别） ==========
        if not hasattr(self.script, '_ore_gained_identified'):
            frame = self.script.current_frame_960x540
            ocr = get_ocr()
            x, y, w, h = 818, 132, 76, 30
            roi = frame[y:y+h, x:x+w]
            text, score = ocr.recognize_single_roi(roi, (0, 0, w, h), preprocess=True)
            ore_count = 0
            if text:
                match = re.search(r'\d+', text)
                if match:
                    ore_count = int(match.group())
            self.script._ore_gained_from_finish = ore_count
            self.script._ore_gained_identified = True   # 标记已识别
    # ====================================================
        
        self._process_step_index += 1

    def _match_get_rewards_page(self, gray, threshold=0.9):
        crop_x, crop_y, crop_w, crop_h = 435, 25, 525, 75
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        res = cv2.matchTemplate(
            crop_gray, self._get_rewards_label_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= threshold

    def _process_steps_2(self):
        current_frame_960x540 = self.script.current_frame_960x540
        gray_frame = cv2.cvtColor(current_frame_960x540, cv2.COLOR_BGR2GRAY)
        if self._match_chatbox(gray=gray_frame, threshold=0.8):
            self._process_step_index += 1
            return
        self.script.macro_text_run("A:0.1", block=True)
        self.time_sleep(0.4)

    def _match_chatbox(self, gray, threshold=0.8) -> bool:
        crop_x, crop_y, crop_w, crop_h = 166, 429, 39, 98
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        res = cv2.matchTemplate(
            crop_gray, self._chatbox_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= threshold