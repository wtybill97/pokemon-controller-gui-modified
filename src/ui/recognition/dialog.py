from PySide6.QtWidgets import QDialog, QLabel, QLineEdit, QGridLayout, QDialogButtonBox, QHBoxLayout, QComboBox, QListWidget
from PySide6.QtGui import QIntValidator, QDoubleValidator, QRegularExpressionValidator, QValidator, QFont
from PySide6.QtCore import QRegularExpression
from PySide6.QtWidgets import QSpacerItem, QSizePolicy


class LaunchRecognitionParasDialog(QDialog):
    def __init__(self, parent, paras: dict):
        super().__init__(parent)
        self._paras = paras.copy()
        self._edit_widgets = dict()
        self._bool_validator = QRegularExpressionValidator(QRegularExpression(
            '^(true|false)$', QRegularExpression.CaseInsensitiveOption))
        self._int_validator = QIntValidator()
        self._double_validator = QDoubleValidator()
        self.setWindowTitle('执行脚本')
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        row = self._add_title_label(0)
        row = self._add_row_spacer(row, 20)
        row = self._add_widgets(row)
        row = self._add_row_spacer(row, 30)
        row = self._add_buttons(row)
        self.adjustSize()
        # +++ 新增：检查是否有Recipe参数，并设置初始状态 +++
        self._setup_recipe_dependencies()

    def _setup_recipe_dependencies(self):
        """设置配方参数的依赖关系"""
        # 获取配方控件
        recipe_widget = self._edit_widgets.get("Recipe")
        if not isinstance(recipe_widget, QComboBox):
            return
        
        # 记录需要根据配方调整的控件
        self._recipe_dependent_widgets = {
            "SparklingPowerLevel": self._edit_widgets.get("SparklingPowerLevel"),
            "CatchingPowerLevel": self._edit_widgets.get("CatchingPowerLevel"),
            "AlphaPowerLevel": self._edit_widgets.get("AlphaPowerLevel"),
            "HumungoPowerLevel": self._edit_widgets.get("HumungoPowerLevel"),
            "TeensyPowerLevel": self._edit_widgets.get("TeensyPowerLevel"),
            "ItemPowerLevel": self._edit_widgets.get("ItemPowerLevel"),
            "BigHaulPowerLevel": self._edit_widgets.get("BigHaulPowerLevel"),
            "CatchPowerRequired": self._edit_widgets.get("CatchPowerRequired"),
        }
        
        # 连接信号槽
        recipe_widget.currentTextChanged.connect(self._on_recipe_changed)
        
        # 初始化状态
        self._on_recipe_changed(recipe_widget.currentText())

    def _on_recipe_changed(self, recipe_name):
        """当配方改变时，自动设置其他参数的建议值"""
        # 配方到参数的映射（作为建议值，不是强制值）
        recipe_suggestions = {
            "闪耀力 - 彩虹 - 1.59%/0.242%": {
                "SparklingPowerLevel": 3,
                "AlphaPowerLevel": 3,
                "CatchingPowerLevel": 3,
                
                "ItemPowerLevel": 0,
                "BigHaulPowerLevel": 0,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "闪耀力 - 混合 - 2.13%/0.118%": {
                "SparklingPowerLevel": 3,
                "AlphaPowerLevel": 3,
                "CatchingPowerLevel": 3,
                
                "ItemPowerLevel": 0,
                "BigHaulPowerLevel": 0,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多2 - 霹霹果 - 0.74%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 2,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多2 - 刺耳果 - 0.74%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 2,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多2 - 洛玫果 - 0.74%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 2,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多2 - 棱瓜果 - 1.4%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 2,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多2 - 草蚕果 - 1.4%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 2,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多2 - 灯浆果 - 1.4%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 2,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多2 - 2灯浆果 - 2.16%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 2,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多2 - 2草蚕果 - 2.16%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 2,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多2 - 2福禄果 - 2.16%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 2,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多2 - 2棱瓜果 - 2.16%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 2,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "闪耀力 - 扁樱果 - 6.2%": {
                "SparklingPowerLevel": 3,
                "AlphaPowerLevel": 3,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 0,
                "BigHaulPowerLevel": 0,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "道具力 - 佛柑果 - 3.1%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 3,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多1 - 1.30%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 1,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多1 - 霹霹果 - 1.30%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 1,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多1 - 刺耳果 - 1.30%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 1,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多1 - 佛柑果 - 1.30%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 1,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多1 - 福禄果 - 1.30%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 1,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果3多多1 - 草蚕果 - 1.30%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 1,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "道具力 - 混合 - 1.17%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 3,
                "BigHaulPowerLevel": 3,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "树果1多多1 - 刺耳果 - 5.56%": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 0,
                
                "ItemPowerLevel": 1,
                "BigHaulPowerLevel": 1,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            "捕获力": {
                "SparklingPowerLevel": 0,
                "AlphaPowerLevel": -1,
                "CatchingPowerLevel": 3,
                
                "ItemPowerLevel": 0,
                "BigHaulPowerLevel": 0,
                "HumungoPowerLevel": -1,
                "TeensyPowerLevel": -1,
            },
            
        }
        
        # 获取当前配方的建议参数设置
        suggestions = recipe_suggestions.get(recipe_name, {})
        
        # 更新每个依赖控件的建议值（但不禁用，允许修改）
        for param_name, widget in self._recipe_dependent_widgets.items():
            if widget is None:
                continue
                
            if param_name in suggestions:
                value = suggestions[param_name]
                
                # 根据控件类型设置建议值
                if isinstance(widget, QLineEdit):
                    # 设置建议值，但允许修改
                    widget.setText(str(value))
                    # +++ 修改：不再禁用编辑 +++
                    # widget.setEnabled(True)  # 保持可编辑
                    widget.setStyleSheet("background-color: #ffffe0;")  # 浅黄色背景表示建议值
                elif isinstance(widget, QComboBox):
                    # 对于布尔值的ComboBox，设置建议选中项
                    if param_name == "CatchPowerRequired":
                        index = widget.findText("true" if value else "false")
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    else:
                        # 其他ComboBox设置建议值
                        index = widget.findText(str(value))
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    # +++ 修改：不再禁用编辑 +++
                    # widget.setEnabled(True)
                    widget.setStyleSheet("background-color: #ffffe0;")
            else:
                # 如果参数不在建议中，清除特殊样式
                widget.setStyleSheet("")  # 清除特殊样式

    def get_paras(self) -> dict:
        paras = self._paras.copy()
        for para in paras.values():
            edit_widgets = self._edit_widgets[para.name]
            if edit_widgets is None:
                continue
            if isinstance(edit_widgets, QLineEdit):
                text = edit_widgets.text()
            elif isinstance(edit_widgets, QComboBox):
                text = edit_widgets.currentText()
            elif isinstance(edit_widgets, QListWidget):
                selected = [item.text() for item in edit_widgets.selectedItems()]
                if not selected and isinstance(para.default_value, list):
                    selected = para.default_value
                para.set_value(selected)
                continue
            if para.value_type == bool:
                if text.lower() == 'true':
                    para.set_value(True)
                else:
                    para.set_value(False)
            elif para.value_type == int:
                para.set_value(int(text))
            elif para.value_type == float:
                para.set_value(float(text))
            elif para.value_type == str:
                para.set_value(text)
        return paras

    def _add_title_label(self, row) -> int:
        title_label = QLabel('参数设置')
        font = QFont()
        font.setPointSize(18)
        title_label.setFont(font)
        self.layout.addWidget(title_label, row, 0, 1, 3)
        row += 1
        return row

    def _add_widgets(self, row) -> int:
        row += 1
        for para in self._paras.values():
            para_summary = para.description
            para_default = para.default_value
            # +++ 修改：为配方参数添加特殊标识 +++
            if para.name == "Recipe":
                para_label = QLabel(f'{para_summary} (配方选择)')
            else:
                para_label = QLabel(f'{para_summary}')
            #para_label = QLabel(f'{para_summary}')
            if para.items and para.value_type == list:
                para_edit = QListWidget()
                para_edit.addItems([str(item) for item in para.items])
                para_edit.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
                if isinstance(para_default, list):
                    for i in range(para_edit.count()):
                        item = para_edit.item(i)
                        if item.text() in para_default:
                            item.setSelected(True)
                if para_edit.count() > 0:
                    row_height = para_edit.sizeHintForRow(0)
                    if row_height <= 0:
                        row_height = para_edit.fontMetrics().height() + 8
                    content_height = row_height * para_edit.count() + para_edit.frameWidth() * 2 + 6
                    para_edit.setMaximumHeight(content_height)
            elif para.items:
                para_edit = QComboBox()
                para_edit.addItems([str(item) for item in para.items])
                para_edit.setCurrentText(str(para_default))
                # +++ 修改：添加提示信息 +++
                if para.name in ["CatchPowerRequired"]:
                    para_edit.setToolTip("当选择特定配方时，此参数会自动设置为建议值，但您可以修改")
            else:
                para_edit = QLineEdit(str(para_default))
                # +++ 修改：为相关参数添加提示 +++
                if para.name in ["SparklingPowerLevel", "ItemPowerLevel", "CatchPowerLevel", 
                              "AlphaPowerLevel", "HumungoPowerLevel", "TeensyPowerLevel", 
                              "BigHaulPowerLevel"]:
                   para_edit.setToolTip("此参数的值会根据所选配方自动设置为建议值，但您可以修改")
            if not isinstance(para_edit, QListWidget):
                self._set_lineEdit_validator(para_edit, para.value_type)
            self.layout.addWidget(para_label, row, 0, 1, 2)
            self.layout.addWidget(para_edit, row, 2, 1, 2)
            self._edit_widgets[para.name] = para_edit
            row += 1
        return row

    def _set_lineEdit_validator(self, widget, value_type: type):
        if value_type == bool:
            widget.setValidator(self._bool_validator)
        elif value_type == int:
            widget.setValidator(self._int_validator)
        elif value_type == float:
            widget.setValidator(self._double_validator)
        elif value_type == str:
            pass
        else:
            raise TypeError("type must be int, float, str or bool")
        return

    def _add_row_spacer(self, row, height=10) -> int:
        spacer = QSpacerItem(
            1, height, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(spacer, row, 0)
        row += 1
        return row

    def _add_buttons(self, row) -> int:
        button_box = QDialogButtonBox()
        button_box.addButton('运行', QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.addButton('取消', QDialogButtonBox.ButtonRole.RejectRole)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.layout.addWidget(button_box, row, 2, 1, 2)
        row += 1
        return row
