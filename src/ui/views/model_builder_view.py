# -*- coding: utf-8 -*-
"""
Model Builder View

Provides visual drag-and-drop neural network model building interface.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSplitter, QTabWidget, QTextEdit, QToolButton, QVBoxLayout, QWidget
)

from src.model_builder.model_graph import ModelGraph
from src.ui.model_editor.layer_palette import LayerPalette
from src.ui.model_editor.layer_property_panel import LayerPropertyPanel
from src.ui.model_editor.model_graph_widget import ModelGraphWidget
from src.ui.i18n import tr_
from src.ui.views.toolbars.model_toolbar import ModelToolbar
from src.ui.styles import Styles
from src.utils.config import config

logger = logging.getLogger(__name__)


@dataclass
class _ModelTab:
    graph_widget: ModelGraphWidget
    model_graph: ModelGraph
    file_path: Optional[str] = None
    source_keras_model: object = None


class ModelBuilderView(QWidget):
    """
    模型构建视图
    
    整合层面板、模型画布和属性面板，提供完整的模型搭建体验。
    """
    
    model_changed = Signal()
    model_saved = Signal(str)  # model_path
    build_requested = Signal()  # 请求构建模型
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._model_graph: Optional[ModelGraph] = None
        self._current_file_path: Optional[str] = None
        self._source_keras_model = None  # 导入的 Keras 模型引用（用于权重迁移）
        self._session_restore_in_progress: bool = True

        # 快照持久化防抖（避免每次拖拽/编辑都写磁盘）
        self._snapshot_timer = QTimer(self)
        self._snapshot_timer.setSingleShot(True)
        self._snapshot_timer.setInterval(800)
        self._snapshot_timer.timeout.connect(self._persist_model_snapshot)
        
        self._setup_ui()
        self._connect_signals()
        self._apply_styles()
        
        # 创建默认模型
        self._create_default_model()

    def set_model_graph(self, model_graph: ModelGraph, file_path: Optional[str] = None):
        """
        外部注入/恢复模型图（用于启动恢复或外部加载）。
        """
        if self._tabs.count() == 0:
            self._add_model_tab(model_graph=model_graph, file_path=file_path, make_current=True)
            return

        tab = self._current_tab()
        if not tab:
            self._add_model_tab(model_graph=model_graph, file_path=file_path, make_current=True)
            return

        tab.model_graph = model_graph
        tab.file_path = file_path
        tab.source_keras_model = None
        tab.graph_widget.set_model_graph(model_graph)
        self._sync_current_model_state()
        self._property_panel.set_layer(None)
        self._property_panel.set_compile_config(model_graph.compile_config)
        self._refresh_all_tab_titles()

    def set_session_restore_in_progress(self, enabled: bool) -> None:
        self._session_restore_in_progress = bool(enabled)

    def persist_session_state(self) -> None:
        self.persist_session_state_guarded()

    def persist_session_state_guarded(self, *, force: bool = False) -> None:
        if self._session_restore_in_progress and not force:
            return

        paths: list[str] = []
        active = -1
        current = self._tabs.currentIndex() if hasattr(self, "_tabs") else -1
        for index in range(self._tabs.count()):
            tab = self._get_tab(index)
            if tab is None or not tab.file_path:
                continue
            path = os.path.abspath(tab.file_path)
            if not os.path.exists(path):
                continue
            if index == current:
                active = len(paths)
            paths.append(path)

        try:
            config.set("session.open_model_paths", paths)
            config.set("session.active_model_tab", active)
        except Exception:
            pass

    def clear_all_tabs(self, *, ensure_one_tab: bool = True) -> None:
        while self._tabs.count() > 0:
            widget = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if widget is not None:
                widget.deleteLater()

        if ensure_one_tab:
            self._add_model_tab(model_graph=ModelGraph("new_model"), file_path=None, make_current=True)
        else:
            self._sync_current_model_state()

        self.persist_session_state_guarded()

    def open_model_file(self, path: str):
        model_graph = ModelGraph.load(path)
        self._add_model_tab(model_graph=model_graph, file_path=path, make_current=True)
        try:
            config.set("session.last_model_path", path)
            config.add_recent_file(path)
        except Exception:
            pass
        self.persist_session_state_guarded()
    
    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._graph_widget: Optional[ModelGraphWidget] = None
        self._create_toolbar(layout)

        self._workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._workspace_splitter.setHandleWidth(2)
        self._workspace_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {Styles.COLORS['surface1']};
            }}
        """)

        self._layer_palette = LayerPalette()
        self._layer_palette.setMinimumWidth(200)
        self._layer_palette.setMaximumWidth(280)
        self._workspace_splitter.addWidget(self._layer_palette)

        self._canvas_column = QWidget()
        canvas_layout = QVBoxLayout(self._canvas_column)
        canvas_layout.setContentsMargins(8, 8, 8, 8)
        canvas_layout.setSpacing(0)
        self._canvas_column.setStyleSheet(f"""
            QWidget {{
                background: {Styles.COLORS['mantle']};
                border: 1px solid {Styles.COLORS['surface1']};
            }}
            QTabWidget, QTabBar, QWidget > QWidget {{
                border: none;
            }}
        """)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setTabsClosable(True)
        self._tabs.setStyleSheet(Styles.canvas_tab_widget())
        self._new_tab_button = QToolButton()
        self._new_tab_button.setText("+")
        self._new_tab_button.setToolTip(tr_("New model"))
        self._new_tab_button.clicked.connect(self._on_new)
        self._tabs.setCornerWidget(self._new_tab_button)
        canvas_layout.addWidget(self._tabs)
        self._workspace_splitter.addWidget(self._canvas_column)

        self._property_panel = LayerPropertyPanel()
        self._property_panel.setMinimumWidth(250)
        self._property_panel.setMaximumWidth(350)
        self._workspace_splitter.addWidget(self._property_panel)
        self._workspace_splitter.setSizes([220, 720, 300])
        self._workspace_splitter.setStretchFactor(0, 0)
        self._workspace_splitter.setStretchFactor(1, 1)
        self._workspace_splitter.setStretchFactor(2, 0)

        layout.addWidget(self._workspace_splitter)
    
    def _create_toolbar(self, layout):
        """创建工具栏"""
        self._toolbar = ModelToolbar(self)
        self._toolbar.new_requested.connect(self._on_new)
        self._toolbar.open_requested.connect(self._on_open)
        self._toolbar.save_requested.connect(self._on_save)
        self._toolbar.save_as_requested.connect(self._on_save_as)
        self._toolbar.copy_requested.connect(self._copy_current_selection)
        self._toolbar.delete_requested.connect(self._delete_current_selection)
        self._toolbar.build_requested.connect(self._on_build)
        self._toolbar.import_keras_requested.connect(self._on_import_keras)
        self._toolbar.fit_requested.connect(self._fit_current)

        self._model_name_label = QLabel(tr_("📐 New model"), self)
        self._model_name_label.hide()
        layout.addWidget(self._toolbar)
    
    def _connect_signals(self):
        """连接信号"""
        self._layer_palette.layer_add_requested.connect(self._on_add_layer)
        self._property_panel.parameter_changed.connect(self._on_parameter_changed)
        self._property_panel.compile_config_changed.connect(self._on_compile_config_changed)
        self._tabs.currentChanged.connect(self._on_current_tab_changed)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
    
    def _create_default_model(self):
        """创建默认模型"""
        self._add_model_tab(model_graph=ModelGraph("new_model"), file_path=None, make_current=True)
        self._property_panel.set_compile_config(self._model_graph.compile_config)
        self._update_title()

    def _add_model_tab(
        self,
        *,
        model_graph: Optional[ModelGraph] = None,
        file_path: Optional[str] = None,
        source_keras_model=None,
        make_current: bool = True,
    ) -> int:
        graph = model_graph or ModelGraph("new_model")
        graph_widget = ModelGraphWidget()
        graph_widget.set_model_graph(graph)
        graph_widget.layer_selected.connect(self._on_layer_selected)
        graph_widget.graph_changed.connect(self._on_graph_changed)

        tab = _ModelTab(
            graph_widget=graph_widget,
            model_graph=graph,
            file_path=file_path,
            source_keras_model=source_keras_model,
        )
        graph_widget.setProperty("model_tab", tab)
        idx = self._tabs.addTab(graph_widget, "")
        self._update_tab_title(idx)

        if make_current:
            self._tabs.setCurrentIndex(idx)
            self._sync_current_model_state()
            self._property_panel.set_layer(None)
            self._property_panel.set_compile_config(graph.compile_config)
            self.model_changed.emit()
            self.persist_session_state_guarded()
        return idx

    def _get_tab(self, index: int) -> Optional[_ModelTab]:
        widget = self._tabs.widget(index)
        if isinstance(widget, ModelGraphWidget):
            tab = widget.property("model_tab")
            if isinstance(tab, _ModelTab):
                return tab
        return None

    def _current_tab(self) -> Optional[_ModelTab]:
        return self._get_tab(self._tabs.currentIndex())

    def _sync_current_model_state(self) -> None:
        tab = self._current_tab()
        if tab is None:
            self._model_graph = None
            self._current_file_path = None
            self._source_keras_model = None
            self._graph_widget = None
            return

        self._model_graph = tab.model_graph
        self._current_file_path = tab.file_path
        self._source_keras_model = tab.source_keras_model
        self._graph_widget = tab.graph_widget

    def _model_display_name(self, model_graph: Optional[ModelGraph], file_path: Optional[str]) -> str:
        if file_path:
            filename = os.path.basename(file_path)
            return filename.replace(".model.json", "").replace(".json", "")
        if model_graph and model_graph.name:
            return model_graph.name
        return "new_model"

    def _tab_title_for_model(self, tab: Optional[_ModelTab]) -> str:
        if tab is None:
            return "new_model"
        title = self._model_display_name(tab.model_graph, tab.file_path)
        if getattr(tab.model_graph, "is_dirty", False):
            return f"{title} *"
        return title

    def _update_tab_title(self, index: int) -> None:
        if 0 <= index < self._tabs.count():
            self._tabs.setTabText(index, self._tab_title_for_model(self._get_tab(index)))

    def _refresh_all_tab_titles(self) -> None:
        for index in range(self._tabs.count()):
            self._update_tab_title(index)
        self._update_title()

    def _is_pristine_empty_tab(self, tab: Optional[_ModelTab]) -> bool:
        if tab is None:
            return False
        graph = tab.model_graph
        return bool(
            not tab.file_path
            and not getattr(graph, "is_dirty", False)
            and not getattr(graph, "layers", {})
            and not getattr(graph, "connections", [])
        )
    
    def _on_add_layer(self, layer_type: str):
        """添加层"""
        self._graph_widget.add_layer(
            layer_type,
            self._graph_widget.get_visible_viewport_center(),
        )
    
    def _on_layer_selected(self, layer_id: str):
        """层选中"""
        if self._model_graph:
            layer = self._model_graph.get_layer(layer_id)
            self._property_panel.set_layer(layer)
    
    def _on_graph_changed(self):
        """模型图变化"""
        tab = self._current_tab()
        if tab is not None:
            tab.model_graph = tab.graph_widget.get_model_graph() or tab.model_graph
        self._update_title()
        self.model_changed.emit()
        self._schedule_persist_snapshot()
    
    def _on_parameter_changed(self, layer_id: str, param_name: str, value):
        """参数变化"""
        # 标记为已修改
        if self._model_graph:
            self._model_graph.is_dirty = True
            self._update_title()
        self._schedule_persist_snapshot()
    
    def _on_compile_config_changed(self):
        """编译配置变化"""
        if self._model_graph:
            self._model_graph.is_dirty = True
            self._update_title()
        self._schedule_persist_snapshot()
    
    def _on_new(self):
        """新建模型"""
        self._add_model_tab(model_graph=ModelGraph("new_model"), file_path=None, make_current=True)
    
    def _on_open(self):
        """打开模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr_("Open model"),
            "models",
            tr_("Model files (*.model.json);;All files (*)")
        )
        
        if file_path:
            try:
                model_graph = ModelGraph.load(file_path)
                current_tab = self._current_tab()
                if self._is_pristine_empty_tab(current_tab) and current_tab is not None:
                    current_tab.model_graph = model_graph
                    current_tab.file_path = file_path
                    current_tab.source_keras_model = None
                    current_tab.graph_widget.set_model_graph(model_graph)
                    self._sync_current_model_state()
                else:
                    self._add_model_tab(model_graph=model_graph, file_path=file_path, make_current=True)
                self._property_panel.set_layer(None)
                self._property_panel.set_compile_config(self._model_graph.compile_config)
                self._refresh_all_tab_titles()
                logger.info(f"模型已加载: {file_path}")

                try:
                    config.set('session.last_model_path', file_path)
                    config.add_recent_file(file_path)
                except Exception:
                    pass
                self.persist_session_state_guarded()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    tr_("Error"),
                    tr_("Failed to load model: {error}").format(error=str(e)),
                )
                logger.exception("加载模型失败")
    
    def _on_import_keras(self):
        """从 Keras 模型文件导入架构"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr_("Import Keras model"),
            "models",
            tr_("Keras models (*.keras *.h5);;All files (*)")
        )
        
        if not file_path:
            return
        
        # 检查当前是否有未保存的更改
        if self._model_graph and self._model_graph.is_dirty:
            reply = QMessageBox.question(
                self,
                tr_("Confirm"),
                tr_("The current model has unsaved changes. Continue importing?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        try:
            import tensorflow as tf
            from src.model_builder.keras_parser import KerasModelParser
            
            # 显示进度提示
            self._model_name_label.setText(tr_("📐 Importing..."))
            
            # 加载 Keras 模型（不编译）
            keras_model = tf.keras.models.load_model(file_path, compile=False)
            
            # 解析为 ModelGraph
            parser = KerasModelParser()
            imported_graph = parser.parse(keras_model)
            tab = self._current_tab()
            if tab is None:
                self._add_model_tab(
                    model_graph=imported_graph,
                    source_keras_model=keras_model,
                    make_current=True,
                )
            else:
                tab.model_graph = imported_graph
                tab.file_path = None
                tab.source_keras_model = keras_model
                tab.graph_widget.set_model_graph(imported_graph)
                self._sync_current_model_state()
            self._property_panel.set_layer(None)
            self._property_panel.set_compile_config(self._model_graph.compile_config)
            self._refresh_all_tab_titles()
            
            # 获取模型摘要
            summary = parser.get_model_summary(keras_model)
            
            # 检查不支持的层
            unsupported = parser.get_unsupported_layers()
            warning_msg = ""
            if unsupported:
                warning_msg = tr_(
                    "\n\n⚠️ The following layer type(s) are not supported and were skipped:\n{layers}"
                ).format(layers=", ".join(unsupported))
            
            # 统计冻结层
            frozen_count = sum(1 for layer in self._model_graph.layers.values() if not layer.trainable)
            frozen_info = (
                tr_("\nFrozen layers: {count}").format(count=frozen_count)
                if frozen_count > 0
                else ""
            )
            
            QMessageBox.information(
                self,
                tr_("Import succeeded"),
                tr_(
                    "Model imported: {name}\n\n"
                    "Layers: {layer_count}\n"
                    "Total params: {total_params:,}\n"
                    "Trainable params: {trainable_params:,}\n"
                    "Non-trainable params: {non_trainable_params:,}"
                ).format(
                    name=summary["name"],
                    layer_count=summary["layer_count"],
                    total_params=summary["total_params"],
                    trainable_params=summary["trainable_params"],
                    non_trainable_params=summary["non_trainable_params"],
                )
                + f"{frozen_info}{warning_msg}",
            )
            
            logger.info(f"Keras模型已导入: {file_path}")
            
        except Exception as e:
            self._update_title()
            QMessageBox.critical(
                self,
                tr_("Import failed"),
                tr_("Failed to import model:\n{error}").format(error=str(e)),
            )
            logger.exception("导入Keras模型失败")
    
    def _on_save(self):
        """保存模型"""
        if not self._model_graph:
            return
        
        if not self._current_file_path:
            # 如果没有当前路径，调用另存为
            self._on_save_as()
            return
        
        self._save_to_path(self._current_file_path)
    
    def _on_save_as(self):
        """模型另存为"""
        if not self._model_graph:
            QMessageBox.warning(self, tr_("Warning"), tr_("No model to save."))
            return
        
        # 默认文件名（确保使用英文名称）
        import re
        default_name = self._model_graph.name or "new_model"
        # 如果名称包含非ASCII字符，使用默认英文名
        if not default_name.isascii():
            default_name = "new_model"
        default_path = f"models/{default_name}.model.json"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            tr_("Save model as"),
            default_path,
            tr_("Model files (*.model.json);;All files (*)")
        )
        
        if not file_path:
            return
        
        # 确保文件扩展名正确
        if not file_path.endswith('.model.json'):
            file_path += '.model.json'
        
        self._save_to_path(file_path)
    
    def _save_to_path(self, file_path: str) -> bool:
        """保存模型到指定路径"""
        try:
            # 确保目录存在
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # 从文件路径中提取模型名称（去掉扩展名）
            file_name = os.path.basename(file_path)
            model_name = file_name.replace('.model.json', '').replace('.json', '')
            
            # 确保模型名称合法（TensorFlow scope 名称规则）
            # 只允许: A-Za-z0-9._\/>-，且不能以特殊字符开头
            import re
            sanitized_name = re.sub(r'[^A-Za-z0-9._/>-]', '_', model_name)
            if sanitized_name and not sanitized_name[0].isalnum() and sanitized_name[0] != '.':
                sanitized_name = 'model_' + sanitized_name
            
            # 更新模型名称
            self._model_graph.name = sanitized_name or 'model'
            
            self._model_graph.save(file_path)
            tab = self._current_tab()
            if tab is not None:
                tab.model_graph = self._model_graph
                tab.file_path = file_path
                tab.source_keras_model = self._source_keras_model
            self._current_file_path = file_path
            self._refresh_all_tab_titles()
            self.model_saved.emit(file_path)

            try:
                config.set('session.last_model_path', file_path)
                config.add_recent_file(file_path)
            except Exception:
                pass
            self.persist_session_state_guarded()
            
            QMessageBox.information(
                self,
                tr_("Saved"),
                tr_("Model saved to:\n{path}").format(path=file_path),
            )
            logger.info(f"模型已保存: {file_path}")
            return True
        except Exception as e:
            QMessageBox.critical(
                self,
                tr_("Error"),
                tr_("Failed to save model: {error}").format(error=str(e)),
            )
            logger.exception("保存模型失败")
            return False
    
    def _on_build(self):
        """构建模型"""
        if not self._model_graph:
            QMessageBox.warning(self, tr_("Warning"), tr_("No model to build."))
            return
        
        # 验证模型
        valid, errors = self._model_graph.validate()
        if not valid:
            QMessageBox.warning(
                self,
                tr_("Validation failed"),
                tr_("Model validation failed:\n") + "\n".join(errors),
            )
            return
        
        try:
            # 构建Keras模型（包含编译）
            keras_model = self._model_graph.build_keras_model(compile_model=True)
            
            # 从输出层获取编译配置信息
            config = self._model_graph.get_compile_config_from_output()
            if config:
                compile_info = (
                    tr_(
                        "Optimizer: {optimizer} (lr={lr})\n"
                        "Loss: {loss}\n"
                        "Metrics: {metrics}"
                    ).format(
                        optimizer=config.get("optimizer", "Adam"),
                        lr=config.get("learning_rate", 0.001),
                        loss=config.get("loss", "mse"),
                        metrics=", ".join(config.get("metrics", ["mae"])),
                    )
                )
            else:
                compile_info = tr_("Using default compile config")
            
            # 显示模型摘要
            summary_lines = []
            keras_model.summary(print_fn=lambda x: summary_lines.append(x))
            summary_text = "\n".join(summary_lines)

            # 创建带滚动条的自定义对话框
            self._show_build_success_dialog(compile_info, summary_text)
            
            # 保存构建的模型
            if self._current_file_path:
                model_save_path = self._current_file_path.replace('.model.json', '.keras')
                keras_model.save(model_save_path)
                logger.info(f"Keras模型已保存: {model_save_path}")
            
            self.build_requested.emit()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                tr_("Build failed"),
                tr_("Model build failed: {error}").format(error=str(e)),
            )
            logger.exception("构建模型失败")
    
    def _on_clear(self):
        """清空画布"""
        if self._model_graph and self._model_graph.layers:
            reply = QMessageBox.question(
                self,
                tr_("Confirm"),
                tr_("Clear all layers?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        self._graph_widget.clear()
        self._model_graph = ModelGraph("new_model")
        self._graph_widget.set_model_graph(self._model_graph)
        tab = self._current_tab()
        if tab is not None:
            tab.model_graph = self._model_graph
            tab.file_path = None
            tab.source_keras_model = None
        self._property_panel.set_layer(None)
        self._property_panel.set_compile_config(self._model_graph.compile_config)
        self._refresh_all_tab_titles()

    def _on_current_tab_changed(self, index: int):
        self._sync_current_model_state()
        self._property_panel.set_layer(None)
        if self._model_graph:
            self._property_panel.set_compile_config(self._model_graph.compile_config)
        self._update_tab_title(index)
        self._update_title()
        self.model_changed.emit()
        self.persist_session_state_guarded()

    def _on_tab_close_requested(self, index: int):
        tab = self._get_tab(index)
        if tab is None:
            return

        if tab.model_graph and getattr(tab.model_graph, "is_dirty", False):
            name = self._model_display_name(tab.model_graph, tab.file_path)
            msg = tr_("Save changes to '{name}' before closing?").format(name=name)
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle(tr_("Confirm"))
            box.setText(msg)
            box.setStandardButtons(
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
            box.setDefaultButton(QMessageBox.StandardButton.Save)
            choice = box.exec()

            if choice == QMessageBox.StandardButton.Cancel:
                return
            if choice == QMessageBox.StandardButton.Save:
                self._tabs.setCurrentIndex(index)
                if not self._save_current_tab():
                    return

        widget = tab.graph_widget
        self._tabs.removeTab(index)
        widget.deleteLater()

        if self._tabs.count() == 0:
            self._add_model_tab(model_graph=ModelGraph("new_model"), file_path=None, make_current=True)
        else:
            self._sync_current_model_state()
            self._property_panel.set_layer(None)
            self._refresh_all_tab_titles()
            self.model_changed.emit()
            self.persist_session_state_guarded()

    def _copy_current_selection(self):
        if self._graph_widget:
            self._graph_widget.copy_selected()

    def _delete_current_selection(self):
        if self._graph_widget:
            self._graph_widget.delete_selected()

    def _fit_current(self):
        if self._graph_widget:
            self._graph_widget.fit_to_selection()

    def _save_current_tab(self) -> bool:
        if not self._model_graph:
            return True
        if self._current_file_path:
            return self._save_to_path(self._current_file_path)
        before = self._current_file_path
        self._on_save_as()
        return bool(self._current_file_path and self._current_file_path != before)
    
    def _update_title(self):
        """更新标题"""
        if self._model_graph:
            dirty_mark = " *" if self._model_graph.is_dirty else ""
            self._model_name_label.setText(f"📐 {self._model_graph.name}{dirty_mark}")
            tab = self._current_tab()
            if tab is not None:
                self._update_tab_title(self._tabs.currentIndex())

    def _schedule_persist_snapshot(self):
        """防抖触发模型快照持久化"""
        if not self._model_graph:
            return
        # 重置定时器
        self._snapshot_timer.start()

    def _persist_model_snapshot(self):
        """将当前模型图快照写入配置（用于启动恢复）"""
        if not self._model_graph:
            return
        try:
            snapshot = self._model_graph.to_dict()
            config.set('session.last_model_graph_snapshot', snapshot)
        except Exception:
            # 快照持久化失败不应影响编辑体验
            pass
    
    def get_model_graph(self) -> Optional[ModelGraph]:
        """获取当前模型图"""
        return self._model_graph
    
    def get_keras_model(self):
        """获取构建的Keras模型"""
        if self._model_graph:
            return self._model_graph.build_keras_model()
        return None
    
    def get_model_file_path(self) -> Optional[str]:
        """获取当前模型文件路径"""
        return self._current_file_path
    
    def _show_build_success_dialog(self, compile_info: str, summary_text: str):
        """显示构建成功对话框（带滚动条的模型架构显示）"""
        dialog = QDialog(self)
        dialog.setWindowTitle(tr_("✅ Model built successfully"))
        dialog.setMinimumSize(700, 500)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # 成功提示
        success_label = QLabel(tr_("🎉 Model built and compiled successfully!"))
        success_label.setStyleSheet(f"""
            QLabel {{
                font-size: 16px;
                font-weight: bold;
                color: {Styles.COLORS['green']};
                padding: 8px;
            }}
        """)
        layout.addWidget(success_label)
        
        # 编译配置区域
        compile_label = QLabel(tr_("Compile config"))
        compile_label.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {Styles.COLORS['mauve']};
            }}
        """)
        layout.addWidget(compile_label)
        
        compile_info_label = QLabel(compile_info)
        compile_info_label.setStyleSheet(f"""
            QLabel {{
                font-size: 13px;
                color: {Styles.COLORS['text']};
                padding: 8px;
                background: {Styles.COLORS['surface0']};
                border-radius: 4px;
            }}
        """)
        compile_info_label.setWordWrap(True)
        layout.addWidget(compile_info_label)
        
        # 模型架构区域
        arch_label = QLabel(tr_("Model architecture"))
        arch_label.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {Styles.COLORS['blue']};
                margin-top: 8px;
            }}
        """)
        layout.addWidget(arch_label)
        
        # 带滚动条的文本框显示模型架构
        summary_text_edit = QTextEdit()
        summary_text_edit.setReadOnly(True)
        summary_text_edit.setPlainText(summary_text)
        summary_text_edit.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Consolas', 'Courier New', 'Monaco', monospace;
                font-size: 12px;
                color: {Styles.COLORS['text']};
                background: {Styles.COLORS['mantle']};
                border: 1px solid {Styles.COLORS['surface1']};
                border-radius: 6px;
                padding: 8px;
            }}
            QScrollBar:vertical {{
                background: {Styles.COLORS['surface0']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {Styles.COLORS['surface2']};
                border-radius: 6px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Styles.COLORS['overlay0']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        layout.addWidget(summary_text_edit, 1)  # stretch factor = 1
        
        # 确定按钮
        ok_btn = QPushButton(tr_("OK"))
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Styles.COLORS['green']};
                color: {Styles.COLORS['crust']};
                border: none;
                border-radius: 6px;
                padding: 10px 32px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Styles.COLORS['teal']};
            }}
        """)
        ok_btn.clicked.connect(dialog.accept)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 设置对话框样式
        dialog.setStyleSheet(f"""
            QDialog {{
                background: {Styles.COLORS['base']};
            }}
        """)
        
        dialog.exec()
    
    def _apply_styles(self):
        """应用样式"""
        self.setStyleSheet(f"""
            ModelBuilderView {{
                background: {Styles.COLORS['base']};
            }}
        """)

