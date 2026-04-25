from enum import Enum
from recognition.scripts.base.base_script import BaseScript
from recognition.scripts.base.base_sub_step import BaseSubStep, SubStepRunningStatus
import cv2
from datetime import datetime
import os

from recognition.scripts.games.pokemon.swsh.common.image_match.pokemon_detail_shiny_match import PokemonDetailShinyMatch


class SWSHDAShinyKeepResult(Enum):
    NotKept = 0
    Kept = 1
    KeptLegendary = 2


class SWSHDAShinyKeep(BaseSubStep):
    def __init__(self, script: BaseScript, legendary_caught: bool, only_keep_shiny_legendary: bool = False, timeout: float = -1) -> None:
        super().__init__(script, timeout)
        self._legendary_caught = legendary_caught
        self._only_keep_shiny_legendary = only_keep_shiny_legendary
        self._process_step_index = 0
        self._check_counter = 0
        self._kept_result = SWSHDAShinyKeepResult.NotKept
        self._keep_pokemon_label_template = cv2.imread(
            "resources/img/recognition/pokemon/swsh/dynamax_adventures/keep_pokemon_label.png")
        self._keep_pokemon_label_template = cv2.cvtColor(
            self._keep_pokemon_label_template, cv2.COLOR_BGR2GRAY)

    @property
    def kept_result(self) -> SWSHDAShinyKeepResult:
        return self._kept_result

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
        ]

    def _process_steps_0(self):
        current_frame_960x540 = self.script.current_frame_960x540
        gray_frame = cv2.cvtColor(current_frame_960x540, cv2.COLOR_BGR2GRAY)
        if not self._match_keep_pokemon_page(gray_frame):
            self.time_sleep(0.5)
            return
        self.script.macro_text_run(
            "TOP:0.1->0.4->A:0.1->0.6->BOTTOM:0.1->0.6->A:0.1", block=True)
        self.time_sleep(2)
        self._check_counter = 0
        self._process_step_index += 1

    # ==================== MODIFIED: 惩罚上限时传说闪光放弃 ====================
    def _process_steps_1(self):
        current_frame_960x540 = self.script.current_frame_960x540
        gray_frame = cv2.cvtColor(current_frame_960x540, cv2.COLOR_BGR2GRAY)
        is_shiny = self._match_shiny(gray_frame)

        # 惩罚上限标志（DYNITE_ORE_PENALTY_LIMIT = 4）
        penalty_limit = self.script._dynite_ore_penalty_count >= 4

        if self._legendary_caught:
            # 传说已捕获的情况
            if is_shiny:
                self.script.macro_text_run("CAPTURE:1", block=True)
                if self._check_counter == 0:
                    # 传说闪光
                    if penalty_limit:
                        # 惩罚上限：放弃传说闪光，不保留，不停止脚本
                        self.script.send_log("极矿石惩罚已达上限，放弃闪光传说宝可梦，继续下一轮")
                        self._send_shiny_notification(is_legendary=True, kept=False)
                        self._quit_pokemon_detail()
                        self._not_keep()  # 放弃
                        
                        self._process_step_index += 1
                        return
                    else:
                        # 正常情况：保留传说闪光
                        self._quit_pokemon_detail()
                        self._keep()  # 内部设置 KeptLegendary
                        self._process_step_index += 1
                        return
                else:
                    # 非传说闪光：记录计数和通知，但不保留
                    self.script._shiny_count += 1
                    self.script.send_log(f"检测到非传说闪光宝可梦（第{self._check_counter+1}只），累计闪光数: {self.script._shiny_count}")
                    self._send_shiny_notification(is_legendary=False, kept=False)
                    self._move_to_next_pokemon()
                    # 修改点：如果已经检查完所有非传说宝可梦（_check_counter >= 3），则结束检查
                    if self._check_counter >= 3:
                        self._quit_pokemon_detail()
                        self._not_keep()
                        self._process_step_index += 1
                    return
            else:
                if self._check_counter >= 3:
                    # 所有宝可梦检查完毕，无传说闪光，放弃所有
                    self._quit_pokemon_detail()
                    self._not_keep()
                    self._process_step_index += 1
                    return
                else:
                    self._move_to_next_pokemon()
                    return
        else:
            # 传说未捕获的情况：遍历所有宝可梦，寻找闪光，保留第一个闪光，否则放弃
            if is_shiny:
                self.script.macro_text_run("CAPTURE:1", block=True)
                # 发现闪光宝可梦（普通宝可梦）
                self.script._shiny_count += 1
                self.script.send_log(
                    f"检测到闪光宝可梦（第{self._check_counter + 1}只），累计闪光数: {self.script._shiny_count}")
                self._send_shiny_notification(is_legendary=False, kept=True)
                # 保留该闪光宝可梦
                self._quit_pokemon_detail()
                self._keep()  # 保留，kept_result = Kept
                self._process_step_index += 1
                return
            else:
                if self._check_counter >= 3:
                    # 所有宝可梦检查完毕，无闪光，放弃所有
                    self._quit_pokemon_detail()
                    self._not_keep()
                    self._process_step_index += 1
                    return
                else:
                    self._move_to_next_pokemon()
                    return
    # ==================== MODIFIED END ====================

    def _move_to_next_pokemon(self):
        self._check_counter += 1
        if self._check_counter < 4:
            self.script.macro_text_run("TOP:0.1", block=True)
            self.time_sleep(0.5)

    def _send_shiny_notification(self, is_legendary: bool = False, kept: bool = False):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"shiny_{timestamp}.png"
            cv2.imwrite(screenshot_path, self.script.current_frame_960x540)
            self.script.send_log(f"闪光宝可梦截图已保存: {screenshot_path}")

            image_key = self.script._upload_feishu_image(screenshot_path)
            if image_key:
                content = {
                    '闪光宝可梦类型': '传说宝可梦' if is_legendary else '非传说宝可梦',
                    '累计闪光数': f"{self.script._shiny_count}/{self.script._total_catch_count}",
                    #'本轮次数': self.script.cycle_times,
                    '成功攻略次数': f"{self.script._win_count}/{self.script.cycle_times}",
                    '累计极矿石': self.script._dynite_ore_total,
                    #'累计捕获宝可梦': self.script._total_catch_count
                }
                title = '✨ 保留并捕获了闪光宝可梦！' if kept else '✨ 检测到闪光宝可梦（未捕获）'
                self.script._send_feishu_card_with_image(
                    title=title,
                    image_key=image_key,
                    content_dict=content
                )
            else:
                self.script._send_feishu_webhook(
                    msg_type='shiny',
                    title='✨ 闪光宝可梦通知',
                    content_dict={
                        '闪光类型': '传说' if is_legendary else '非传说',
                        '累计闪光数': self.script._shiny_count,
                        '本轮次数': self.script.cycle_times,
                        '详情': '请手动确认宝可梦（图片上传失败）'
                    }
                )
            os.remove(screenshot_path)
        except Exception as e:
            self.script.send_log(f"发送闪光通知失败: {e}")

    def _quit_pokemon_detail(self):
        self.script.macro_text_run("B:0.1", block=True)
        self.time_sleep(2)

    def _keep(self):
        if self._check_counter == 0 and self._legendary_caught:
            self._kept_result = SWSHDAShinyKeepResult.KeptLegendary
        else:
            self.script.macro_text_run(
                "A:0.05->0.5->A:0.05->0.5->A:0.05->0.5->A:0.05->0.5->A:0.05", block=True)
            self.time_sleep(1)
            self._kept_result = SWSHDAShinyKeepResult.Kept

    def _not_keep(self):
        self.script.macro_text_run(
            "B:0.1->0.6->A:0.1->0.4->A:0.1->0.4->A:0.1", block=True)
        self.time_sleep(1)

    def _match_keep_pokemon_page(self, gray, threshold=0.9) -> bool:
        crop_x, crop_y, crop_w, crop_h = 435, 25, 525, 75
        crop_gray = gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        res = cv2.matchTemplate(
            crop_gray, self._keep_pokemon_label_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val >= threshold

    def _match_shiny(self, gray, threshold=0.9) -> bool:
        return PokemonDetailShinyMatch().match_shiny(gray=gray, threshold=threshold)