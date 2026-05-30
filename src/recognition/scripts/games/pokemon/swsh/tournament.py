# -*- coding: utf-8 -*-
"""
宝可梦-剑盾-淘汰赛自动化脚本（带文字统计）
支持配置 loop（循环次数）和 durations（运行时长，分钟）
新增：每轮输出本轮奖励，每20轮输出累计统计，一轮超时20分钟自动停止
"""

import time
import cv2
import numpy as np
import pytesseract
import re
from recognition.scripts.base.base_script import BaseScript, WorkflowEnum
from recognition.scripts.base.base_sub_step import BaseSubStep, SubStepRunningStatus
from recognition.scripts.parameter_struct import ScriptParameter
from recognition.image_func import find_matches
from recognition.ocr.rapidocr import RapidOCR

# ---------- 通用 OCR 引擎（单例） ----------
_shared_ocr = None

def get_ocr():
    global _shared_ocr
    if _shared_ocr is None:
        _shared_ocr = RapidOCR(upscale=3.0, enable_preprocess=True)
    return _shared_ocr


# ---------- 初始步骤 ----------
class EliminationStart(BaseSubStep):
    def __init__(self, script: BaseScript, timeout: float = -1):
        super().__init__(script, timeout)
        self._process_step_index = 0

    def _process(self):
        if self._process_step_index >= len(self._process_steps):
            return SubStepRunningStatus.OK
        self._process_steps[self._process_step_index]()
        return SubStepRunningStatus.Running

    @property
    def _process_steps(self):
        return [self._step0]

    def _step0(self):
        macro_cmd = "LSTICK@0,-127:2->0.8->A:0.1->0.8->A:0.1->0.8->A:0.1->0.8->A:0.1->0.8->B:0.1->0.8->A:0.1->0.8->A:0.1->A:0.1->0.8"
        try:
            self.script.macro_text_run(macro_cmd, block=True)
        except Exception as e:
            self.script.send_log(f"初始宏执行异常: {e}")
        self._process_step_index += 1


# ---------- 战斗招式选择 ----------
# ---------- 战斗招式选择（修改后） ----------
class EliminationBattle(BaseSubStep):
    def __init__(self, script: BaseScript, timeout: float = 30, avoid_current: bool = False):
        super().__init__(script, timeout)
        self._move_pp_history = {}
        self._choose_move_arrow_template = self._load_template(
            "resources/img/recognition/pokemon/swsh/dynamax_adventures/battle/choose_move_arrow.png"
        )
        self._step = 0
        self._battle_start_time = None
        self._avoid_current = avoid_current

    def _load_template(self, path):
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError
            return img
        except:
            return np.zeros((10, 10), dtype=np.uint8)

    def _process(self):
        if self._battle_start_time is None:
            self._battle_start_time = time.monotonic()
        if self._timeout > 0 and time.monotonic() - self._battle_start_time > self._timeout:
            self.script.send_log("战斗招式选择超时，跳过")
            return SubStepRunningStatus.OK

        frame = self.script.current_frame_960x540
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._step == 0:
            arrow_roi = gray[320:320+220, 620:620+46]
            matches = find_matches(arrow_roi, self._choose_move_arrow_template, threshold=0.7, min_distance=10)
            if not matches:
                return SubStepRunningStatus.Running
            self._step = 1
            self._current_index = self._get_move_index(matches[0][1])
            return SubStepRunningStatus.Running

        elif self._step == 1:
            ocr = get_ocr()

            # 1. 先只识别当前招式的 PP 和效果
            current_pp = self._ocr_pp(gray, 867, 334 + 52 * self._current_index, 70, 26, move_index=self._current_index)

            effect_roi = gray[351 + 53 * self._current_index:351 + 53 * self._current_index + 18, 689:689 + 54]
            effect_text, _ = ocr.recognize_single_roi(
                cv2.cvtColor(effect_roi, cv2.COLOR_GRAY2BGR),
                (0, 0, 54, 18), preprocess=True
            )
            effect_text = effect_text or ""
            if "效果绝佳" in effect_text or "效果佳" in effect_text:
                cur_effect = "效果佳"
            elif "有效果" in effect_text and effect_text != "有效果":
                cur_effect = "没有效果"
            elif "效果不好" in effect_text:
                cur_effect = "效果不好"
            else:
                cur_effect = "有效果"

            # 2. 如果不需要避开当前技能，且当前 PP>0，且效果为效果佳或有效果，则直接使用
            if not self._avoid_current and current_pp > 0 and cur_effect in ("效果佳", "有效果"):
                target = self._current_index
                #self.script.send_log(f"当前技能{cur_effect}，直接使用")
                self.script.macro_text_run("A:0.1->0.6", block=True, loop=2)
                if current_pp > 0:
                    self._move_pp_history[target] = current_pp - 1
                return SubStepRunningStatus.OK

            # 3. 否则，完整识别所有招式的效果和 PP
            effects = []
            pps = []
            for i in range(4):
                # 效果
                effect_roi = gray[351 + 53 * i:351 + 53 * i + 18, 689:689 + 54]
                effect_text, _ = ocr.recognize_single_roi(
                    cv2.cvtColor(effect_roi, cv2.COLOR_GRAY2BGR),
                    (0, 0, 54, 18), preprocess=True
                )
                effect_text = effect_text or ""
                if "效果绝佳" in effect_text or "效果佳" in effect_text:
                    effect = "效果佳"
                elif "有效果" in effect_text and effect_text != "有效果":
                    effect = "没有效果"
                elif "效果不好" in effect_text:
                    effect = "效果不好"
                else:
                    effect = "有效果"
                effects.append(effect)

                # PP
                pp = self._ocr_pp(gray, 867, 334 + 52 * i, 70, 26, move_index=i)
                pps.append(pp)

            # 构建候选列表
            candidates = [(i, effects[i], pps[i]) for i in range(4) if pps[i] > 0]
            if not candidates:
                self.script.send_log("战斗：没有可用招式")
                return SubStepRunningStatus.OK

            from collections import defaultdict
            groups = defaultdict(list)
            for idx, eff, _ in candidates:
                if eff == "效果佳":
                    prio = 0
                elif eff == "有效果":
                    prio = 1
                elif eff == "效果不好":
                    prio = 2
                else:
                    prio = 3
                groups[prio].append(idx)
            for prio in groups:
                groups[prio].sort(key=lambda idx: abs(idx - self._current_index))

            target = None
            for prio in range(4):
                if groups[prio]:
                    target = groups[prio][0]
                    break
            if target is None:
                self.script.send_log("战斗：没有可用招式")
                return SubStepRunningStatus.OK

            # 定身法处理
            if self._avoid_current and target == self._current_index:
                if effects[target] == "效果佳":
                    current_prio = 0
                elif effects[target] == "有效果":
                    current_prio = 1
                elif effects[target] == "效果不好":
                    current_prio = 2
                else:
                    current_prio = 3
                if len(groups[current_prio]) > 1:
                    target = groups[current_prio][1]
                else:
                    for prio in range(current_prio + 1, 4):
                        if groups[prio]:
                            target = groups[prio][0]
                            break

            # 移动光标，步数 3 时反向移动一步
            move = target - self._current_index
            if abs(move) == 3:
                move = -1 if move > 0 else 1

            if move < 0:
                self.script.macro_text_run("TOP:0.05->0.1", block=True, loop=abs(move))
            elif move > 0:
                self.script.macro_text_run("BOTTOM:0.05->0.25", block=True, loop=move)

            self.script.macro_text_run("A:0.1->0.6", block=True, loop=2)

            if pps[target] > 0:
                self._move_pp_history[target] = pps[target] - 1

            return SubStepRunningStatus.OK

        return SubStepRunningStatus.OK

    def _get_move_index(self, arrow_y):
        if arrow_y < 30:
            return 0
        elif arrow_y < 80:
            return 1
        elif arrow_y < 130:
            return 2
        else:
            return 3

    def _ocr_pp(self, gray, x, y, w, h, move_index, zoom=5):
        """
        识别 PP 数值，支持格式 "数字/数字"，提取斜杠前的数字。
        如果识别失败，返回 0。
        同时保留历史纠错：若识别为0且上一轮真实PP>2，则纠正为上一轮PP-1。
        """
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (w*zoom, h*zoom))
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/'
        text = pytesseract.image_to_string(roi, config=custom_config)
        text = "".join(text.split())
        # 提取斜杠前的数字
        match = re.search(r'^(\d+)/', text)
        if match:
            ocr_pp = int(match.group(1))
        else:
            # 兼容旧格式（纯数字）
            if text.isdigit():
                ocr_pp = int(text)
            else:
                ocr_pp = 0

        # 纠错逻辑（保留）
        prev_pp = self._move_pp_history.get(move_index)
        if ocr_pp == 0 and prev_pp is not None and prev_pp > 2:
            corrected_pp = prev_pp - 1
            return corrected_pp
        return ocr_pp


# ---------- 移动摇杆 ----------
class EliminationMove(BaseSubStep):
    def __init__(self, script: BaseScript, timeout: float = 10):
        super().__init__(script, timeout)
        self._done = False

    def _process(self):
        if self._done:
            return SubStepRunningStatus.OK
        self.script.macro_text_run("LSTICK@0,-127:2->1", block=True)
        self._done = True
        return SubStepRunningStatus.Running


# ---------- 奖励处理（含统计和完成计数，新增背包满专用区域识别） ----------
class EliminationReward(BaseSubStep):
    TARGET_BALLS = ["究极球", "速度球", "等级球", "沉重球", "诱饵球", "月亮球", "甜蜜球", "友友球"]

    def __init__(self, script: BaseScript, timeout: float = 30):
        super().__init__(script, timeout)
        self._process_step_index = 0
        self._reward_text = ""

    def _process(self):
        if self._process_step_index >= len(self._process_steps):
            return SubStepRunningStatus.OK
        self._process_steps[self._process_step_index]()
        return SubStepRunningStatus.Running

    @property
    def _process_steps(self):
        return [self._step0, self._step1]

    def _step0(self):
        frame = self.script.current_frame_960x540
        ocr = get_ocr()
        roi_main = frame[477:477+42, 199:199+328]
        text_main, _ = ocr.recognize_single_roi(roi_main, (0, 0, 328, 42), preprocess=True)
        if text_main:
            if "数量已满" in text_main:
                roi_full = frame[439:439+40, 203:203+300]
                full_text, _ = ocr.recognize_single_roi(roi_full, (0, 0, 300, 40), preprocess=True)
                if full_text:
                    # 提取第一个标点符号之后、“的”之前的文字
                    match = re.search(r'[！？。，；：！“”‘’、…—～·（）【】《》](.*?)的', full_text)
                    if match:
                        item_name = match.group(1).strip()
                    else:
                        idx = full_text.find("的")
                        if idx != -1:
                            item_name = full_text[idx+1:].strip()
                        else:
                            item_name = full_text.strip()
                    cleaned = re.sub(r'[^\w\u4e00-\u9fff]', '', item_name)
                    if cleaned:
                        self._reward_text = cleaned
                        self.script._current_round_reward = cleaned  # 记录本轮奖励
                        self._update_statistics(cleaned)
                        self._process_step_index += 1
                        return
                    else:
                        self.script.send_log("背包满区域未识别到有效物品名称")
                else:
                    self.script.send_log("背包满区域未识别到文字")
                self._reward_text = "背包已满"
                self.script._current_round_reward = None  # 无实际物品
                self._process_step_index += 1
                return
            elif "获得了" in text_main:
                self._reward_text = text_main
                # 提取“获得了”之后的文字作为本轮奖励
                idx = text_main.find("获得了")
                if idx != -1:
                    after_text = text_main[idx+3:]
                    cleaned = re.sub(r'[^\w\u4e00-\u9fff]', '', after_text)
                    self.script._current_round_reward = cleaned if cleaned else None
                else:
                    self.script._current_round_reward = None
                self._update_statistics(text_main)
                self._process_step_index += 1
                return
            else:
                self._process_step_index += 1
                return
        else:
            self.script.send_log("奖励区域未识别到文字")
            self._process_step_index += 1

    def _step1(self):
        found_balls = [ball for ball in self.TARGET_BALLS if ball in self._reward_text]
        if found_balls:
            self.script.send_log(f"获得目标球种: {', '.join(found_balls)}")
            self._send_notification_with_stats(found_balls)
        self.script.macro_text_run("A:0.1->1->A:0.1->1", block=True, loop=4)
        time.sleep(0.5)
        self.script._complete_round()
        self._process_step_index += 1

    def _update_statistics(self, full_text):
        if isinstance(full_text, str) and "获得了" in full_text:
            idx = full_text.find("获得了")
            if idx != -1:
                after_text = full_text[idx + 3:]
            else:
                after_text = full_text
        else:
            after_text = full_text
        cleaned = re.sub(r'[^\w\u4e00-\u9fff]', '', after_text)
        if not cleaned:
            return
        if not hasattr(self.script, '_reward_stats'):
            self.script._reward_stats = {}
        self.script._reward_stats[cleaned] = self.script._reward_stats.get(cleaned, 0) + 1

    def _get_rare_ball_total(self):
        """计算累计获得的珍稀球种总次数（从统计字典中筛选）"""
        if not hasattr(self.script, '_reward_stats'):
            return 0
        total = 0
        for ball in self.TARGET_BALLS:
            total += self.script._reward_stats.get(ball, 0)
        return total

    def _send_notification_with_stats(self, found_balls):
        stats_str = self._get_stats_string()
        rare_total = self._get_rare_ball_total()
        current_round = self.script._completed_rounds + 1  # 本轮结束后的轮数
        feishu_content = {
            "details": f"奖励球种: {', '.join(found_balls)}\n\n【文字出现次数统计】\n{stats_str}\n\n珍稀球种累计次数 / 当前轮次: {rare_total} / {current_round}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        meow_content = f"奖励球种: {', '.join(found_balls)}\n{stats_str}\n珍稀球种累计次数 / 当前轮次: {rare_total} / {current_round}"
        self.script.send_notification(
            title="🎉 淘汰赛获得珍贵球种",
            feishu_content=feishu_content,
            meow_title="淘汰赛获得珍贵球种",
            meow_content=meow_content
        )

    def _get_stats_string(self):
        if not hasattr(self.script, '_reward_stats') or not self.script._reward_stats:
            return "暂无统计"
        lines = []
        for text, count in sorted(self.script._reward_stats.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{text}: {count}次")
        return "\n".join(lines)


# ---------- 循环检测主步骤 ----------
# ---------- 循环检测主步骤（修改版：入口匹配需持续0.2秒） ----------
class EliminationLoop(BaseSubStep):
    def __init__(self, script: BaseScript, timeout: float = -1):
        super().__init__(script, timeout)
        self._sub_state = 0
        self._battle_sub = None
        self._reward_sub = None
        self._move_sub = None
        self._entrance_template = None
        self._no_action_counter = 0
        self._battle_interval_counter = 0
        self._max_no_action = 60
        self._gift_sub_state = 0
        self._gift_wait_start = 0
        self._gift_timeout = 10
        # 入口连续匹配计时
        self._entrance_match_start = 0
        self._entrance_match_confirm = False

    def _load_entrance_template(self):
        if self._entrance_template is None:
            path = "resources/img/recognition/pokemon/swsh/entrance.jpg"
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.ones((50, 50), dtype=np.uint8) * 255
            self._entrance_template = img
        return self._entrance_template

    def _process(self):
        if self._sub_state == 0:
            frame = self.script.current_frame_960x540
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            matched = False

            # 入口匹配（需要连续0.2秒）
            if self._match_entrance(gray):
                now = time.monotonic()
                if self._entrance_match_start == 0:
                    self._entrance_match_start = now
                elif now - self._entrance_match_start >= 0.2 and not self._entrance_match_confirm:
                    # 已达到0.2秒，触发移动
                    self._entrance_match_confirm = True
                    self._sub_state = 3
                    self._move_sub = EliminationMove(self.script)
                    matched = True
                    self._entrance_match_start = 0  # 重置
                else:
                    # 还在等待连续匹配中，不触发其他匹配，直接返回
                    return SubStepRunningStatus.Running
            else:
                # 匹配失败，重置计时器
                if self._entrance_match_start != 0:
                    self._entrance_match_start = 0
                    self._entrance_match_confirm = False

            # 如果入口匹配未触发（即未达到0.2秒且当前帧匹配成功），则继续等待，不进行后续检测
            # 注意：只有入口匹配未触发时，才允许进行战斗/奖励匹配
            if not matched and self._entrance_match_start == 0:
                # 战斗文字检测
                ocr = get_ocr()
                battle_roi = frame[318:318+28, 792:792+48]
                text, _ = ocr.recognize_single_roi(battle_roi, (0, 0, 48, 28), preprocess=True)
                if text and "战斗" in text:
                    interval = self._battle_interval_counter
                    avoid = (0 < interval < 3)
                    if avoid:
                        self.script.send_log(f"检测到定身法（上次战斗后仅{interval}次无操作），将避开当前箭头技能")
                    self._battle_interval_counter = 0
                    self.script.macro_text_run("A:0.1", block=True)
                    self.time_sleep(0.5)
                    self._sub_state = 1
                    self._battle_sub = EliminationBattle(self.script, timeout=30, avoid_current=avoid)
                    matched = True

            if not matched and self._entrance_match_start == 0:
                # 奖励文字检测
                ocr = get_ocr()
                reward_roi = frame[477:477+42, 199:199+328]
                reward_text, _ = ocr.recognize_single_roi(reward_roi, (0, 0, 328, 42), preprocess=True)
                if reward_text:
                    if "获得了" in reward_text or "数量已满" in reward_text:
                        self._sub_state = 2
                        self._reward_sub = EliminationReward(self.script, timeout=30)
                        matched = True
                    elif "这个礼物" in reward_text:
                        self.script.macro_text_run("B:0.1->1->B:0.1->1", block=True)
                        self._gift_sub_state = 1
                        self._gift_wait_start = time.monotonic()
                        self._sub_state = 4
                        matched = True

            if matched:
                self._no_action_counter = 0
            else:
                # 只有未触发任何匹配时才执行按 B
                if self._entrance_match_start == 0:
                    self.script.macro_text_run("B:0.1->0.9", block=True)
                    self.time_sleep(0.1)
                    self._no_action_counter += 1
                    self._battle_interval_counter += 1
                    if self._no_action_counter >= self._max_no_action:
                        self.script.send_notification(
                            title="⚠️ 淘汰赛脚本异常停止",
                            feishu_content={"details": "连续60次无有效操作"},
                            meow_title="淘汰赛脚本异常",
                            meow_content="连续60次无有效操作"
                        )
                        self.script.stop_work()
                        return SubStepRunningStatus.OK
            return SubStepRunningStatus.Running

        # 其余子状态（1,2,3,4）与原代码相同，保持不变
        elif self._sub_state == 1:
            if self._battle_sub is None:
                self._sub_state = 0
                return SubStepRunningStatus.Running
            status = self._battle_sub.run()
            if status == SubStepRunningStatus.OK:
                self._sub_state = 0
                self._battle_sub = None
                self._no_action_counter = 0
            return status

        elif self._sub_state == 2:
            if self._reward_sub is None:
                self._sub_state = 0
                return SubStepRunningStatus.Running
            status = self._reward_sub.run()
            if status == SubStepRunningStatus.OK:
                self._sub_state = 0
                self._reward_sub = None
                self._no_action_counter = 0
                self._battle_interval_counter = 0
                self.script._round_finished = True
            return status

        elif self._sub_state == 3:
            if self._move_sub is None:
                self._sub_state = 0
                return SubStepRunningStatus.Running
            status = self._move_sub.run()
            if status == SubStepRunningStatus.OK:
                self._sub_state = 0
                self._move_sub = None
                self._no_action_counter = 0
                self._battle_interval_counter = 0
            return status

        elif self._sub_state == 4:
            if time.monotonic() - self._gift_wait_start > self._gift_timeout:
                self.script.send_log("等待真实奖励超时（10秒），脚本停止")
                self.script.send_notification(
                    title="⚠️ 淘汰赛脚本异常停止",
                    feishu_content={"details": "等待真实奖励超时（10秒）"},
                    meow_title="淘汰赛脚本异常",
                    meow_content="等待真实奖励超时（10秒）"
                )
                self.script.stop_work()
                return SubStepRunningStatus.OK
            frame = self.script.current_frame_960x540
            ocr = get_ocr()
            reward_roi = frame[477:477+42, 199:199+328]
            reward_text, _ = ocr.recognize_single_roi(reward_roi, (0, 0, 328, 42), preprocess=True)
            if reward_text and ("获得了" in reward_text or "数量已满" in reward_text):
                self._sub_state = 2
                self._reward_sub = EliminationReward(self.script, timeout=30)
                self._gift_sub_state = 0
                return SubStepRunningStatus.Running
            return SubStepRunningStatus.Running

        return SubStepRunningStatus.Running

    def _match_entrance(self, gray, threshold=0.8):
        template = self._load_entrance_template()
        x, y = 440, 165
        w, h = 48, 28
        x_end = min(x + w, gray.shape[1])
        y_end = min(y + h, gray.shape[0])
        roi = gray[y:y_end, x:x_end]
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= threshold


# ---------- 主脚本类 ----------
class SwshEliminationTournament(BaseScript):
    @staticmethod
    def script_name() -> str:
        return "宝可梦-剑盾-淘汰赛"

    @staticmethod
    def script_paras() -> dict:
        paras = dict()
        paras["loop"] = ScriptParameter(
            "loop", int, -1, "运行次数（-1表示无限）")
        paras["durations"] = ScriptParameter(
            "durations", float, -1, "运行时长（分钟，-1表示无限制）")
        return paras

    def __init__(self, stop_event, frame_queue, controller_input_action_queue, paras=None):
        super().__init__(SwshEliminationTournament.script_name(),
                         stop_event, frame_queue, controller_input_action_queue,
                         paras if paras is not None else SwshEliminationTournament.script_paras())
        self._prepare_step_index = -1
        self._cycle_step_index = -1
        self._round_finished = False
        self._reward_stats = {}

        self._loop = self.get_para("loop")
        self._durations = self.get_para("durations")

        self._elimination_start = None
        self._completed_rounds = 0

        # 新增：记录每轮开始时间及本轮奖励
        self._round_start_time = None
        self._current_round_reward = None

    def _complete_round(self):
        """完成一轮时调用，增加计数、输出本轮奖励、检查超时，每20轮输出累计统计"""
        # 检查本轮是否超时（20分钟）
        if self._round_start_time is not None:
            round_duration = time.monotonic() - self._round_start_time
            if round_duration > 20 * 60:  # 20分钟 = 1200秒
                self.send_log(f"本轮耗时 {round_duration:.1f} 秒，超过20分钟，脚本停止")
                self._finished_process(error_msg=f"一轮耗时超过20分钟（{round_duration:.1f}秒）")
                return

        self._completed_rounds += 1
        run_time_span = self.run_time_span
        run_time_str = f"{int(run_time_span/3600)}小时{int((run_time_span%3600)/60)}分{int(run_time_span%60)}秒"

        # 输出本轮获得的奖励
        reward_text = self._current_round_reward if self._current_round_reward else "无奖励"
        self.send_log(f"完成第 {self._completed_rounds} 轮，运行总时间: {run_time_str}，本轮获得: {reward_text}")

        # 每20轮输出累计统计信息
        if self._completed_rounds % 20 == 0:
            if hasattr(self, '_reward_stats') and self._reward_stats:
                stats_lines = ["【累计统计信息（每20轮）】"]
                for text, count in sorted(self._reward_stats.items(), key=lambda x: x[1], reverse=True):
                    stats_lines.append(f"  {text}: {count}次")
                self.send_log("\n".join(stats_lines))
            else:
                self.send_log("累计统计信息为空")

        # 检查是否达到次数限制
        if self._loop > 0 and self._completed_rounds >= self._loop:
            self.send_log("运行次数已到达设定值，脚本停止")
            self._finished_process()

    def _check_durations(self):
        if self._durations <= 0:
            return False
        if self.run_time_span >= self._durations * 60:
            self.send_log("运行时间已到达设定值，脚本停止")
            self._finished_process()
            return True
        return False

    def _check_cycles(self):
        if self._loop <= 0:
            return False
        if self._completed_rounds >= self._loop:
            return True
        return False

    def _send_final_notification(self):
        """发送最终统计通知（脚本停止时）"""
        if hasattr(self, '_reward_stats') and self._reward_stats:
            # 计算珍稀球种总次数
            rare_total = 0
            target_balls = ["究极球", "速度球", "等级球", "沉重球", "诱饵球", "月亮球", "甜蜜球", "友友球"]
            for ball in target_balls:
                rare_total += self._reward_stats.get(ball, 0)
            # 构建统计字符串
            stats_lines = []
            for text, count in sorted(self._reward_stats.items(), key=lambda x: x[1], reverse=True):
                stats_lines.append(f"{text}: {count}次")
            stats_str = "\n".join(stats_lines)
            feishu_content = {
                "details": f"【最终统计】各文字出现次数：\n{stats_str}\n\n珍稀球种累计次数: {rare_total}\n完成总轮数: {self._completed_rounds}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            meow_content = f"【最终统计】各文字出现次数：\n{stats_str}\n珍稀球种累计次数: {rare_total}\n完成总轮数: {self._completed_rounds}"
            self.send_notification(
                title="📊 淘汰赛脚本结束统计",
                feishu_content=feishu_content,
                meow_title="淘汰赛脚本结束统计",
                meow_content=meow_content
            )
        else:
            # 没有统计信息时也发送一个简单通知
            self.send_notification(
                title="📊 淘汰赛脚本结束统计",
                feishu_content={"details": "未获得任何奖励文字"},
                meow_title="淘汰赛脚本结束统计",
                meow_content="未获得任何奖励文字"
            )

    def _finished_process(self, error_msg: str = None):
        if error_msg:
            self.send_notification(
                title="⚠️ 淘汰赛脚本异常停止",
                feishu_content={"details": error_msg},
                meow_title="淘汰赛脚本异常",
                meow_content=error_msg
            )
        # 发送最终统计通知（无论是否异常）
        self._send_final_notification()
        run_time_span = self.run_time_span
        self.macro_stop(block=True)
        self.send_log("[{}] 脚本完成，已完成{}轮，耗时{}小时{}分{}秒".format(
            SwshEliminationTournament.script_name(),
            self._completed_rounds,
            int(run_time_span / 3600),
            int((run_time_span % 3600) / 60),
            int(run_time_span % 60)
        ))
        self.stop_work()

    def process_frame(self):
        if self._check_durations():
            return
        if self._check_cycles():
            return

        if self.running_status == WorkflowEnum.Preparation:
            if self._prepare_step_index >= 0:
                if self._prepare_step_index >= len(self._prepare_step_list):
                    self.set_cycle_begin()
                    self._cycle_step_index = 0
                    self._round_finished = False
                    return
                self._prepare_step_list[self._prepare_step_index]()
            return

        if self.running_status == WorkflowEnum.Cycle:
            if self.current_frame_count == 1:
                self._cycle_init()
            if self._cycle_step_index >= 0 and self._cycle_step_index < len(self._cycle_step_list):
                self._cycle_step_list[self._cycle_step_index]()
            else:
                self.macro_stop()
                self.set_cycle_continue()
                self._cycle_step_index = 0
            return

        if self.running_status == WorkflowEnum.AfterCycle:
            self.stop_work()
            return

    def on_start(self):
        self._prepare_step_index = 0
        self.send_log(f"开始运行{SwshEliminationTournament.script_name()}脚本")
        if self._loop > 0:
            self.send_log(f"目标完成次数: {self._loop} 轮")
        if self._durations > 0:
            self.send_log(f"运行时长限制: {self._durations} 分钟")

    def on_cycle(self):
        pass

    def on_stop(self):
        if hasattr(self, '_reward_stats') and self._reward_stats:
            summary = ["【最终统计】各文字出现次数："]
            for text, count in sorted(self._reward_stats.items(), key=lambda x: x[1], reverse=True):
                summary.append(f"{text}: {count}次")
            self.send_log("\n".join(summary))
        self.send_log(f"脚本停止，共完成 {self._completed_rounds} 轮")

    def on_error(self):
        pass

    @property
    def _prepare_step_list(self):
        return [self._prepare_step0]

    def _prepare_step0(self):
        if self._elimination_start is None:
            self._elimination_start = EliminationStart(self)
        status = self._elimination_start.run()
        if status == SubStepRunningStatus.OK:
            self._prepare_step_index += 1
            self._elimination_start = None

    @property
    def _cycle_step_list(self):
        return [self._cycle_step0]

    def _cycle_init(self):
        # 记录本轮开始时间
        self._round_start_time = time.monotonic()
        self._current_round_reward = None
        self._elimination_loop = EliminationLoop(self)

    def _cycle_step0(self):
        status = self._elimination_loop.run()
        if status == SubStepRunningStatus.OK:
            self.macro_stop(block=True)
            if self._round_finished:
                time.sleep(1.0)
                while not self._controller_input_action_queue.empty():
                    try:
                        self._controller_input_action_queue.get_nowait()
                    except:
                        pass
                self.macro_text_run("LSTICK@0,-127:2->0.8->A:0.1->0.8->A:0.1->0.8->A:0.1->0.8->A:0.1->0.8->B:0.1->0.8->A:0.1->0.8->A:0.1->A:0.1->0.8", block=True)
                self._elimination_loop = EliminationLoop(self)
                self._cycle_step_index = 0
                self._round_finished = False
            else:
                self.set_cycle_continue()
                self._cycle_step_index = 0