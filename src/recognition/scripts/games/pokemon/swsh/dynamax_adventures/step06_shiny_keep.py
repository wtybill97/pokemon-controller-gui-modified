# step06_shiny_keep.py - 完整修改版
from enum import Enum
from recognition.scripts.base.base_script import BaseScript
from recognition.scripts.base.base_sub_step import BaseSubStep, SubStepRunningStatus
import cv2
from datetime import datetime
import os
from recognition.ocr.rapidocr import RapidOCR
from recognition.scripts.games.pokemon.swsh.common.image_match.pokemon_detail_shiny_match import PokemonDetailShinyMatch


class SWSHDAShinyKeepResult(Enum):
    NotKept = 0
    Kept = 1
    KeptLegendary = 2


class SWSHDAShinyKeep(BaseSubStep):
    _rapid_ocr = None

    @classmethod
    def _get_ocr(cls):
        if cls._rapid_ocr is None:
            cls._rapid_ocr = RapidOCR(upscale=3.0, enable_preprocess=True)
        return cls._rapid_ocr

    def __init__(self, script: BaseScript, legendary_caught: bool, only_keep_shiny_legendary: bool = False, 
                 keep_all_shiny: bool = False, used_ball: str = "", timeout: float = -1) -> None:
        super().__init__(script, timeout)
        self._legendary_caught = legendary_caught
        self._only_keep_shiny_legendary = only_keep_shiny_legendary
        self._keep_all_shiny = keep_all_shiny
        self._used_ball = used_ball
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
                    pokemon_name = self._ocr_pokemon_name(current_frame_960x540)
                    if penalty_limit:
                        self.script.send_log("极矿石惩罚已达上限，放弃闪光传说宝可梦，继续下一轮")
                        self._send_shiny_notification(is_legendary=True, kept=False, pokemon_name=pokemon_name) 
                        self._quit_pokemon_detail()
                        self._not_keep()  # 放弃
                        self._process_step_index += 1
                        return
                    else:
                        self._send_shiny_notification(is_legendary=True, kept=True, pokemon_name=pokemon_name)
                        self._quit_pokemon_detail()
                        self._keep()  # 内部设置 KeptLegendary
                        self.script._save_captured_shiny(pokemon_name, self._used_ball) 
                        self._process_step_index += 1
                        return
                else:
                    # 非传说闪光
                    pokemon_name = self._ocr_pokemon_name(current_frame_960x540)
                    self.script._shiny_count += 1
                    self.script.send_log(f"检测到非传说闪光宝可梦（第{self._check_counter+1}只），累计闪光数: {self.script._shiny_count}")
                    if self._keep_all_shiny:
                        self._send_shiny_notification(is_legendary=False, kept=True, pokemon_name=pokemon_name)
                        self._quit_pokemon_detail()
                        self._keep()   # 保留该非传说闪光
                        self.script._save_captured_shiny(pokemon_name, self._used_ball) 
                        self._process_step_index += 1
                        return
                    else:
                        self._send_shiny_notification(is_legendary=False, kept=False, pokemon_name=pokemon_name)
                        self._move_to_next_pokemon()
                        if self._check_counter >= 3:
                            self._quit_pokemon_detail()
                            self._not_keep()
                            self._process_step_index += 1
                        return
            else:
                if self._check_counter >= 3:
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
                pokemon_name = self._ocr_pokemon_name(current_frame_960x540)
                self.script._shiny_count += 1
                self.script.send_log(
                    f"检测到闪光宝可梦（第{self._check_counter + 1}只），累计闪光数: {self.script._shiny_count}")
                self._send_shiny_notification(is_legendary=False, kept=True, pokemon_name=pokemon_name)
                self._quit_pokemon_detail()
                self._keep()
                self.script._save_captured_shiny(pokemon_name, self._used_ball) 
                self._process_step_index += 1
                return
            else:
                if self._check_counter >= 3:
                    self._quit_pokemon_detail()
                    self._not_keep()
                    self._process_step_index += 1
                    return
                else:
                    self._move_to_next_pokemon()
                    return

    def _move_to_next_pokemon(self):
        self._check_counter += 1
        if self._check_counter < 4:
            self.script.macro_text_run("TOP:0.1", block=True)
            self.time_sleep(0.5)

    def _ocr_pokemon_name(self, frame_bgr) -> str:
        x, y, w, h = 530, 22, 180, 42
        results = self._get_ocr().batch_recognize_regions(frame_bgr, [(x, y, w, h)])
        if results and len(results) > 0:
            text_obj = results[0]
            if text_obj and isinstance(text_obj, dict):
                recognized_text = text_obj.get('text', "")
                if recognized_text is None:
                    recognized_text = ""
                recognized_text = recognized_text.strip()
                if recognized_text:
                    name = "".join(recognized_text.split())
                    name = name.replace(" ", "").replace("　", "")
                    return name if name else "未知"
        return "未知"
    
    def _send_shiny_notification(self, is_legendary: bool = False, kept: bool = False, pokemon_name: str = ""):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"shiny_{timestamp}.png"
            cv2.imwrite(screenshot_path, self.script.current_frame_960x540)
            self.script.send_log(f"闪光宝可梦截图已保存: {screenshot_path}")

            feishu_content = {
                '闪光宝可梦类型': '传说宝可梦' if is_legendary else '非传说宝可梦',
                #'宝可梦名称': pokemon_name,
                '累计闪光数': f"{self.script._shiny_count}/{self.script._total_catch_count}",
                '成功攻略次数': f"{self.script._win_count}/{self.script.cycle_times}",
                '累计极矿石': self.script._dynite_ore_total,
            }
            title = f'✨ 保留并捕获了闪光{pokemon_name}！' if kept else f'✨ 检测到闪光{pokemon_name}（未捕获）'
            meow_content = f"宝可梦：{pokemon_name}\n累计闪光数：{self.script._shiny_count}/{self.script._total_catch_count}\n成功攻略次数：{self.script._win_count}/{self.script.cycle_times}\n累计极矿石：{self.script._dynite_ore_total}"
            self.script.send_notification(
                title=title,
                feishu_content=feishu_content,
                image_path=screenshot_path,
                meow_title=title,
                meow_content=meow_content
            )
            self.script._trigger_obs_save('shiny_notification', is_legendary=is_legendary, kept=kept)
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
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= threshold

    def _match_shiny(self, gray, threshold=0.9) -> bool:
        return PokemonDetailShinyMatch().match_shiny(gray=gray, threshold=threshold)