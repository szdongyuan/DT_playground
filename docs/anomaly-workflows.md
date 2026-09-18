# 原生无监督异常工作流

关联：[Issue #154](https://github.com/szdongyuan/DT_playground/issues/154)。GUI 与 CLI 使用相同节点实现，仅新增“特征标准化”和“异常评估”两个节点。

## 节点与连接

训练数据只能包含预期正常的参考样本。标签只连接到评估节点，不参与特征标准化、AE 优化、近邻参考库或阈值标定。

| 能力 | 节点及参数 |
| --- | --- |
| 拟合预处理 | `feature_standardization`: `mode=fit_transform`；输出 `features`、`state` |
| 复用预处理 | 同节点 `mode=transform`，必须输入训练时的 `state` |
| 窗口化 | `feature_vectorizer`: `mode=sliding_window`, `window_length`, `window_stride` |
| kNN | `anomaly_detector_trainer`: `algorithm=knn`, `n_neighbors`, `distance` |
| AE 训练 | `regression_trainer`，同一矩阵同时接 `x_train/y_train`，验证矩阵同时接 `x_val/y_val` |
| AE 封装 | `save_anomaly_model`: `mode=autoencoder`；输入网络、正常参考矩阵、预处理状态 |
| 推理 | `load_anomaly_model` 输出模型与预处理状态，复用相同窗口参数后连接 `anomaly_scorer` |
| 文件聚合 | 评分节点 `aggregation=none/mean/max/top_fraction_mean`，`top_fraction=0.1` |
| 评估 | `anomaly_evaluation`: `protocol=generic/dcase`，输入分数或决策结果，以及 ID→0/1 标签映射 |

时序标准化针对最后一维以外的每个位置（例如每通道每 Mel 频带），使用训练文件全部原始帧拟合均值和总体标准差，然后才切窗口。矩阵标准化按列拟合。常量维的尺度下限由 `epsilon` 控制。已标准化的输入不得再次拟合；经典检测器须设置 `scaling=none`。

窗口默认 5 帧、步长 5，沿最后一维截取，`flatten` 按 C 顺序展开。`tail_policy=include_last` 增加一个覆盖末尾的完整窗口，可能与前一窗口重叠；`drop` 丢弃不足一个窗口的尾部。短文件默认报错，也可选择 `pad_edge`，补帧数写入来源记录。时间边界按特征帧步长计算，不表示 FFT 窗口覆盖范围。

每行保留唯一 ID 和原文件 ID；窗口明细还包含窗口序号、帧边界、补帧数及时间边界。同名不同目录的文件不会按 basename 合并。无来源路径的数组使用顺序 ID，跨工作流复用时应由调用方保证身份稳定。类型化训练 X/Y 必须具有相同的行 ID 顺序，训练与验证契约必须相同且父样本不得重叠。

## 评分语义

kNN 为精确距离的 k 邻居平均值，支持欧氏距离和余弦距离。默认 `reference_exclusion=parent`，查询时排除同一原文件的全部窗口；`sample` 仅排除相同样本 ID。值相同但 ID 不同的样本仍是有效邻居。余弦距离拒绝零向量；排除后邻居不足会报错，不自动减小 k。可连接独立正常 `calibration_features`，其文件不能与训练库重叠。精确搜索按批次计算，窗口库很大时计算量仍然显著。

AE 输入和输出必须都是 `(batch, feature_count)`，评分为逐行重构 MSE，批量推理不启用训练模式。评分器输出 `detailed_scores` 明细与 `anomaly_scores` 最终分数。聚合先使用原始分数，再基于同样聚合后的正常参考分数重算 1%/99% 显示界限；最高比例取 `ceil(窗口数 × top_fraction)` 个窗口。显示尺度是 0–100，不是异常概率。评估和预览排序使用未裁剪的原始分数。

AE 封装需要显式正常参考数据，保存其原始分数与父 ID，因此加载后仍可切换文件聚合方式。参考分数不是独立评估成绩，不能用开发测试标签选择阈值后宣称无监督验证。

通用评估给出 ROC AUC 和标准化 pAUC（默认 `max_fpr=0.1`）。DCASE 模式按 section 分组，每个域 AUC 使用该域正常文件及该 section 所有域的异常文件；pAUC 使用整个 section，再对这些指标取调和平均。缺少类别、域或身份映射会报错。输入决策结果时另给出 precision、recall、F1、误报率、漏检率和混淆矩阵。

标签映射必须完整、唯一且与输出 sample ID 完全一致，不进行模糊 basename 匹配。可用已有 `target_file` 的 `target_map` 输出；设置 `target_kind=continuous,dtype=int64`，标签为正常 0、异常 1。DCASE section/domain 可从规范文件名提取，或由程序调用提供结构化标签记录。文件级评估拒绝未聚合的窗口分数。

## CLI

所有命令在仓库根目录执行，使用项目虚拟环境。

```powershell
# kNN 正常数据训练，模板已包含标准化、窗口化、保存和分数导出
.\.venv\Scripts\python.exe cli_main.py create-anomaly-workflow train.workflow.json --dataset D:\data\normal_fit --algorithm knn --k 5
.\.venv\Scripts\python.exe cli_main.py run train.workflow.json --run-dir outputs\knn_train

# AE：先用 build-model 将已有模型定义构建为编译后的 .keras 网络
.\.venv\Scripts\python.exe cli_main.py build-model model.json --output initial.keras
.\.venv\Scripts\python.exe cli_main.py create-anomaly-workflow ae.workflow.json --dataset D:\data\normal_fit --validation-dataset D:\data\normal_val --algorithm autoencoder --model initial.keras --epochs 40
.\.venv\Scripts\python.exe cli_main.py run ae.workflow.json --run-dir outputs\ae_train

# 对明确可信的本地模型进行测试及 DCASE 评估
.\.venv\Scripts\python.exe cli_main.py create-anomaly-workflow score.workflow.json --phase score --dataset D:\data\test --model outputs\ae_train\anomaly_model.anomaly.zip --trust-model --labels test_labels.json --protocol dcase
.\.venv\Scripts\python.exe cli_main.py validate score.workflow.json --check-data --json
.\.venv\Scripts\python.exe cli_main.py run score.workflow.json --run-dir outputs\ae_test
```

模板固定采用 16 kHz、64 Mel、FFT 1024、hop 512、逐文件峰值 dB、5 帧窗口和 320 维向量；AE 网络须匹配 320 维输入/输出并使用 MSE。要改提取、窗口或网络参数，编辑生成的 JSON 或在 GUI 修改节点。训练与正常验证目录不能相等或互为父子目录；跨目录复制的相同内容仍需在数据准备时去重。

导出目录包含完整 CSV 和元数据；明细分数导出包含窗口来源列。评估指标和训练恢复信息记录在运行 manifest 的节点输出中。`validate` 检查结构、参数及数据源；实际特征 schema 和邻居数量还需在执行时检查。

## 模型兼容与训练恢复

v1 孤立森林包继续可读，对旧包只校验它实际保存的特征字段；旧包不能获得其未记录的完整预处理保证。新训练的模型保存提取布局和标准化签名。kNN、AE 或携带外部标准化状态的模型使用 v2，网络独立保存为 `network.keras`；载荷包含哈希并检查运行版本、归档成员和大小。保存拒绝覆盖，加载仍要求明确可信授权，哈希不等于安全沙箱。

异常包用于推理，加载 Keras 时不恢复编译和优化器，不能作为续训检查点。普通训练器输出的最佳模型现在同步恢复最佳 epoch 的优化器状态；保存普通 Keras 模型时附带 `.training.json`，训练摘要也由端口输出。停止并保存检查点记录停止时最后 epoch 的网络和优化器。恢复级别为 `model_optimizer`，不保存完整随机数、输入流水线或早停回调历史，不保证逐位连续训练等价；epoch 为本次运行的相对计数。

## 本地开发验证

使用已有 Valve seed 42 延长训练权重，不额外训练大模型，对 1,200 文件、75,600 窗口执行原生提取、标准化、窗口化、打包、重载及评分。与旧脚本文件分数相比，平均 MSE 最大差异约 `3.7e-8`，最高 10% 均值最大差异约 `6.0e-8`，通过 `atol=1e-6, rtol=1e-5`。缓存特征有约 `3.6e-6` 的最大差异，不宣称特征逐元素完全一致。该数据已用于多轮开发比较，不是独立泛化评估。
