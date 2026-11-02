# 本地资料与备份索引

## 2026-11-02 本次整理

当前项目根目录保留生产源码、选定模型、测试、配置、依赖与现行说明。开发材料完整移动到[本地归档目录](tmp/development_cleanup_20261102_154735/)，没有删除原始材料或覆盖此前备份；tmp由Git和Docker忽略。以下链接仅供本地使用。

| 整理前 | 当前本地位置 |
|---|---|
| `training/`（工具、样本、训练结果及回放报告） | `tmp/development_cleanup_20261102_154735/training/` |
| `Lamps_labeling/`（标注原图与poly） | `tmp/development_cleanup_20261102_154735/Lamps_labeling/` |
| `Ultralytics/`、`output/`（库配置及缓存） | 归档目录下同名子目录 |
| `IMPLEMENTATION_STATUS.md`、`RELEASE_REVIEW_20261102.md` | 归档目录的`development_notes/` |
| 生产接线文档的完整历史记录 | `development_notes/PRODUCTION_WIRING_before_cleanup.md` |
| `deploy/`内原有`__pycache__/` | 归档目录的`runtime_caches/`，按原相对位置保留 |

移动清单：[move_plan.json](tmp/development_cleanup_20261102_154735/move_plan.json)；校验记录：[MOVE_VERIFIED.json](tmp/development_cleanup_20261102_154735/MOVE_VERIFIED.json)。逐项核对文件数量、大小、修改时间、文件ID和相对路径，生产运行源码保持原样。

- [最新15分钟完整回放、结果图和07:00模拟](tmp/development_cleanup_20261102_154735/training/lamps/pulse_first_20261102/index.html)
- [人工灯具标注目录](tmp/development_cleanup_20261102_154735/Lamps_labeling/)
- 原始双路15分钟录像：`tmp/lamp_night_job_20261030/recording/`，未移动。
- 早期拉流材料：`tmp/development_archive_20261027/tmp/yolo错误模型/tmp/yolo_samples/`，未移动。
- 先前完成版备份：`tmp/backups/completed_lamp_v7_20261102_150436/`及`completed_lamp_v7_20261102_151848/`，未改动。
- 整理后完成版备份：`tmp/backups/organized_completed_v7_20261102_*/`，附`MANIFEST.json`和`RESTORE.md`。
- 整理后检查及上传记录：`tmp/organized_release_20261102/`。

离线脚本和样本中的历史绝对路径按原样保留。后续再次训练时，先恢复或重配离线开发根目录；不要直接将归档当成已重定位的训练环境。生产运行不读取该归档。

本地回归若需要对照归档中的离线算法，可设置测试专用环境变量`BARRIGUARD_OFFLINE_TEST_DIR`为归档的`training/lamps`绝对路径，再运行`python -B -m unittest discover -s deploy/tests -v`。此开关只在测试中读取，服务器没有离线资料时会跳过对应可选对照用例。

## 早期归档记录

以下为2026-10-27整理及更早完成版的历史索引，本次未再次移动这些tmp目录。

开发资料完整保留在`tmp/development_archive_20261027/`，没有删除，由Git及Docker忽略规则排除。

| 原位置 | 当前归档位置 |
|---|---|
| `training/` | `tmp/development_archive_20261027/training/` |
| `output/`历史回放、检查图及日志 | `tmp/development_archive_20261027/output/` |
| `tools/`标注、训练、开发工具 | `tmp/development_archive_20261027/tools/` |
| `docs/`开发与模型报告 | `tmp/development_archive_20261027/docs/` |
| `models/`旧预训练权重 | `tmp/development_archive_20261027/models/` |
| `deploy/tests/` | `tmp/development_archive_20261027/deploy/tests/` |
| `deploy/app/data/`旧日志 | `tmp/development_archive_20261027/deploy/app/data/` |
| 旧blocks标定及灯光历史版本 | 归档`deploy/app/roi_configs/`相同相对位置 |
| 原`tmp/`中的旧模型、视频及回归基线 | `tmp/development_archive_20261027/tmp/` |
| `Ultralytics/`、`__pycache__/` | 归档中相同相对位置 |

整理前的运行源码、配置、README和pyproject也有快照。精确记录见[move_plan.json](tmp/development_archive_20261027/move_plan.json)。

常用本地入口：

- [最新功能优化报告](tmp/development_archive_20261027/docs/WATER_YOLO_OPTIMIZATION_REPORT.md)
- [最新录像结果](tmp/development_archive_20261027/output/water_policy_optimized_review_20261027/index.html)
- [V6模型分析](tmp/development_archive_20261027/docs/V6_OPTIMIZATION_REPORT.md)
- [V7模型分析](tmp/development_archive_20261027/docs/V7_OPTIMIZATION_REPORT.md)

历史训练清单中的绝对路径未改写，以保留审计记录。以后训练前，按移动记录恢复开发布局或生成新的路径清单，不要直接将归档当成已重定位的训练环境。

此外不提交：`.venv/`、服务器`deploy/.env`与`config.local.yaml`、运行生成的`data/`与`output/`、临时检查副本。本次检查记录在`tmp/deployment_tests/`，不提交GitHub。

2026-11-02完成版备份位于`tmp/backups/completed_lamp_v7_*`，含逐文件SHA256和恢复说明；`Lamps_labeling/`、训练视频、训练结果与本地验收视频均被Git和Docker忽略。本轮发布检查记录在`tmp/completion_release_20261102/`。

没有生成或上传压缩包。

2026-10-28：旧灯光源码和原 circle/poly 标注素材已从生产目录移入 tmp/retired_lamp_backend_*。对应 move_manifest.json 记录移动位置；原 tmp/backups/before_lamp_refactor_* 备份未修改。
