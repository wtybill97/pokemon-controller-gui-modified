import time
from enum import Enum
from recognition.image_func import find_matches
from recognition.scripts.base.base_script import BaseScript
from recognition.scripts.base.base_sub_step import BaseSubStep, SubStepRunningStatus
import cv2
import re
import numpy as np
import pytesseract
from recognition.ocr.rapidocr import RapidOCR

from recognition.scripts.games.pokemon.swsh.common.image_match.checkbox_match import ChatBoxMatch

class SWSHDABattleResult(Enum):
    Error = -1
    Won = 0
    Lost1 = 1
    Lost2 = 2
    Lost3 = 3
    Running = 9


class SWSHDABattle(BaseSubStep):
    _rapid_ocr = None

    @classmethod
    def _get_ocr(cls):
        if cls._rapid_ocr is None:
            cls._rapid_ocr = RapidOCR(upscale=3.0, enable_preprocess=True)
        return cls._rapid_ocr

    def __init__(self, script: BaseScript, battle_index: int = 0, timeout: float = -1, disable_dynamax=False) -> None:
        super().__init__(script, timeout)
        self._process_step_index = 0
        self._battle_index = battle_index
        self._disable_dynamax = disable_dynamax
        self._battle_status = 0
        self._last_action_time_monotonic = time.monotonic()
        self._pokemon_name_cached = None
        self._action_template = cv2.imread(
            "resources/img/recognition/pokemon/swsh/dynamax_adventures/battle/action.png")
        self._action_template = cv2.cvtColor(self._action_template, cv2.COLOR_BGR2GRAY)
        self._dynamax_icon_template = cv2.imread(
            "resources/img/recognition/pokemon/swsh/dynamax_adventures/battle/dynamax_icon.png")
        self._dynamax_icon_template = cv2.cvtColor(self._dynamax_icon_template, cv2.COLOR_BGR2GRAY)
        self._choose_move_arrow_template = cv2.imread(
            "resources/img/recognition/pokemon/swsh/dynamax_adventures/battle/choose_move_arrow.png")
        self._choose_move_arrow_template = cv2.cvtColor(self._choose_move_arrow_template, cv2.COLOR_BGR2GRAY)
        self._won_template = cv2.imread(
            "resources/img/recognition/pokemon/swsh/dynamax_adventures/battle/won.png")
        self._won_template = cv2.cvtColor(self._won_template, cv2.COLOR_BGR2GRAY)
        self._lost_1_template = cv2.imread(
            "resources/img/recognition/pokemon/swsh/dynamax_adventures/keep_pokemon_label.png")
        self._lost_1_template = cv2.cvtColor(self._lost_1_template, cv2.COLOR_BGR2GRAY)
        self._lost_2_template = cv2.imread(
            "resources/img/recognition/pokemon/swsh/dynamax_adventures/get_rewards_label.png")
        self._lost_2_template = cv2.cvtColor(self._lost_2_template, cv2.COLOR_BGR2GRAY)
        # ADDED: 历史PP记录字典
        self._move_pp_history = {}  # {index: last_actual_pp}

    @property
    def battle_status(self) -> SWSHDABattleResult:
        return self._battle_status

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
        ]

    def _process_steps_0(self):
        if time.monotonic() - self._last_action_time_monotonic > 120:
            self._status = SubStepRunningStatus.Timeout
            self._battle_status = SWSHDABattleResult.Error
            return

        current_frame_960x540 = self.script.current_frame_960x540
        gray_frame = cv2.cvtColor(current_frame_960x540, cv2.COLOR_BGR2GRAY)

        if self._match_action(gray_frame):
            return
        elif self._match_dynamax_icon(gray_frame):
            return
        elif self._choose_move(gray_frame):
            return
        elif self._match_won(gray_frame):
            self._process_step_index += 1
            self._battle_status = SWSHDABattleResult.Won
            self.time_sleep(1)
            return
        elif self._match_lost_1(gray_frame):
            self._process_step_index += 1
            self._battle_status = SWSHDABattleResult.Lost1
            return
        elif self._match_lost_2(gray_frame):
            self._process_step_index += 1
            self._battle_status = SWSHDABattleResult.Lost2
            return
        elif self._match_lost_3(gray_frame):
            self._process_step_index += 1
            self._battle_status = SWSHDABattleResult.Lost3
            return
        else:
            self.time_sleep(1)

    def _match_action(self, gray, threshold=0.9):
        crop_x, crop_y, crop_w, crop_h = 887, 421, 66, 110
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        res = cv2.matchTemplate(crop_gray, self._action_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            # 第一次匹配到 action.png 时识别宝可梦名称并缓存
            if self._pokemon_name_cached is None:
                name = self.get_pokemon_name()
                if name:
                    self._pokemon_name_cached = name
                    self.script._current_pokemon_name = name
            self.script.macro_text_run("A:0.1", block=True)
            self._last_action_time_monotonic = time.monotonic()
            self.time_sleep(0.5)
            return True
        return False

    def _match_dynamax_icon(self, gray, threshold=0.8):
        if self._disable_dynamax:
            return False
        crop_x, crop_y, crop_w, crop_h = 522, 424, 52, 32
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        res = cv2.matchTemplate(crop_gray, self._dynamax_icon_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            self.script.macro_text_run("LEFT:0.1->0.4->A:0.05->0.1->A:0.05", block=True)
            self._last_action_time_monotonic = time.monotonic()
            self.time_sleep(0.5)
            return True
        return False

    def _choose_move(self, gray):
        crop_x, crop_y, crop_w, crop_h = 620, 320, 46, 220
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        matches = find_matches(crop_gray, self._choose_move_arrow_template, threshold=0.7, min_distance=10)
        if len(matches) == 0:
            return False

        current_index = self._get_current_move(matches[0][1])
        if current_index < 0:
            return False

        ocr = self._get_ocr()
        effects = []
        pps = []

        for i in range(4):
            effect_crop_x, effect_crop_y = 689, 351 + 53 * i
            effect_w, effect_h = 54, 18
            roi_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            roi_effect = roi_bgr[effect_crop_y:effect_crop_y+effect_h,
                                 effect_crop_x:effect_crop_x+effect_w]
            text_raw, score, _ = ocr.recognize_single_roi(
                roi_effect, (0, 0, effect_w, effect_h), preprocess=True, return_raw=True)
            if text_raw is None:
                effect_text = ""
            else:
                effect_text = "".join(text_raw.split())

            if "效果绝佳" in effect_text or "效果佳" in effect_text:
                effect = "效果佳"
            elif "有效果" in effect_text and effect_text != "有效果":
                #self.script.send_log(f"识别到效果技能为{effect_text}")
                #print(effect_text)
                effect = "没有效果"
            elif "效果不好" in effect_text:
                effect = "效果不好"
            elif "没有效果" in effect_text:
                effect = "没有效果"
            else:
                effect = effect_text

            # MODIFIED: 调用修改后的 _ocr_move_pp，传入索引 i
            pp = self._ocr_move_pp(gray, 872, 335 + 53 * i, 24, 27, move_index=i, zoom=5)
            effects.append(effect)
            pps.append(pp)

        # 选择优先级
        choice_first = -1
        choice_second = -1
        choice_third = -1
        choice_forth = -1
        for i in range(4):
            if pps[i] <= 0:
                continue
            if effects[i] == "效果佳":
                choice_first = i
                break
            if effects[i] == "有效果" and choice_second < 0:
                choice_second = i
            if effects[i] == "效果不好" and choice_third < 0:
                choice_third = i
            if choice_forth < 0:
                choice_forth = i

        target = -1
        if choice_first >= 0:
            target = choice_first
        elif choice_second >= 0:
            target = choice_second
        elif choice_third >= 0:
            target = choice_third
        else:
            target = choice_forth

        if target < 0:
            self.script.send_log("[战斗] 未找到可用技能（可能全部PP为0）")
            return False

        # 移动光标
        move_times = target - current_index
        if move_times < 0:
            self.script.macro_text_run("TOP:0.05->0.1", block=True, loop=abs(move_times))
        elif move_times > 0:
            self.script.macro_text_run("BOTTOM:0.05->0.25", block=True, loop=move_times)

        # 确认选择技能
        self.script.macro_text_run("A:0.1->0.6", block=True, loop=6)
        self.script.macro_text_run("BOTTOM:0.1->0.3->A:0.1->0.3", block=True)
        self.script.macro_text_run("RIGHT:0.1->0.3->A:0.1->0.3", block=True, loop=2)
        self.time_sleep(1)
        self._last_action_time_monotonic = time.monotonic()

        # ADDED: 更新该招式的历史PP（使用后剩余PP = 当前PP - 1）
        current_pp = pps[target]  # 已经经过纠错
        if current_pp > 0:
            new_pp = current_pp - 1
            self._move_pp_history[target] = new_pp
        else:
            self._move_pp_history[target] = 0

        return True

    def _match_won(self, gray, threshold=0.9):
        crop_x, crop_y, crop_w, crop_h = 748, 445, 204, 90
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        res = cv2.matchTemplate(crop_gray, self._won_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            self._last_action_time_monotonic = time.monotonic()
            return True
        return False

    def _match_lost_1(self, gray, threshold=0.9):
        crop_x, crop_y, crop_w, crop_h = 435, 25, 525, 75
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        res = cv2.matchTemplate(crop_gray, self._lost_1_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            self._last_action_time_monotonic = time.monotonic()
            return True
        return False

    def _match_lost_2(self, gray, threshold=0.9):
        crop_x, crop_y, crop_w, crop_h = 435, 25, 525, 75
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        res = cv2.matchTemplate(crop_gray, self._lost_2_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            self._last_action_time_monotonic = time.monotonic()
            return True
        return False

    def _match_lost_3(self, gray, threshold=0.9):
        if ChatBoxMatch().match_next_arrow(gray=gray, threshold=threshold):
            self._last_action_time_monotonic = time.monotonic()
            return True
        return False

    def _get_current_move(self, y):
        if 0 <= y < 30:
            return 0
        elif 53 <= y < 83:
            return 1
        elif 106 <= y < 136:
            return 2
        elif 159 <= y < 189:
            return 3
        else:
            return -1

    # MODIFIED: 增加 move_index 参数，实现基于历史记录的纠错（差值判断）
    def _ocr_move_pp(self, gray, crop_x, crop_y, crop_w, crop_h, move_index, zoom=5):
        """
        使用 Tesseract 识别 PP，并基于历史记录纠错：
        若识别为0且上一轮该招式实际PP - 0 > 2，则纠正为上一轮PP-1；
        否则保持识别值。
        """
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        crop_gray = cv2.resize(crop_gray, (crop_w*zoom, crop_h*zoom))
        _, thresh1 = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        closing = cv2.resize(closing, (crop_w, crop_h))
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(closing, config=custom_config)
        text = "".join(text.split())
        ocr_pp = int(text) if text.isdigit() else 0

        # ADDED: 纠错逻辑
        prev_pp = self._move_pp_history.get(move_index)
        if ocr_pp == 0 and prev_pp is not None:
            diff = prev_pp - ocr_pp  # 即 prev_pp
            if diff > 3:
                # 误识别（如6/9识别成0），纠正为上一轮PP-1
                corrected_pp = prev_pp - 1
                #self.script.send_log(f"[PP纠错] 招式{move_index} OCR识别为0，历史PP={prev_pp}（差值{diff}>2），纠正为{corrected_pp}")
                return corrected_pp
            # 否则保持0（正常消耗，如上一轮1或2）
        return ocr_pp

    def get_pokemon_name(self):
        """识别画面中宝可梦名称（区域 x=350, y=12, w=250, h=52）"""
        try:
            frame = self.script.current_frame_960x540
            ocr = self._get_ocr()  # 复用 RapidOCR 实例
            x, y, w, h = 350, 12, 250, 52
            roi = frame[y:y+h, x:x+w]
            text, _ = ocr.recognize_single_roi(roi, (0, 0, w, h), preprocess=True)
            if text:
                # 去除标点符号和空格
                import re
                name = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
                self.script.send_log(f"识别到宝可梦名称：{name}")
                return name
        except Exception as e:
            self.script.send_log(f"识别宝可梦名称失败：{e}")
        return None