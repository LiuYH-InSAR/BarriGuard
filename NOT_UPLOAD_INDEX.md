# 不提交GitHub的本地索引

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

没有生成或上传压缩包。
