# 上传索引

服务器通过 Git 获取源码，再构建容器。

当前上传目标为`origin/work/current-water-lamp`（本地`main`的既有上游），保留旧`master`。完成版复核见`RELEASE_REVIEW_20261102.md`，上传不自动启用服务器灯具。

| 路径 | 用途 |
|---|---|
| deploy/app/camera_motion/ | 公共偏移、参考图库、几何变换与迁移 |
| deploy/app/water_barrier/ | 现有 V6 水马功能 |
| deploy/app/night_lamp/ | 唯一 YOLO 灯具方案：定位、身份、观察、报告、保存 |
| deploy/app/server/ | 配置、API、拉流、调度与回调 |
| deploy/config.yaml、config.test.yaml | 验收前保持灯具禁用的模板 |
| deploy/Dockerfile、docker-compose.yml | 运行环境与持久化 |
| deploy/docker-compose.lamp.yml | 可选覆盖内置 v7 灯具模型的只读挂载 |
| deploy/requirements-server.txt、.env.example | 依赖与配置变量示例 |
| models/water_barrier_v6/ | 已选水马模型及校验信息 |
| models/lamp_v7/ | 已选灯具 v7 权重、320输入配置与校验信息 |
| deploy/tests/ | 水马、共享模块和新灯具回归 |
| check_deploy.py、deploy_manifest.json | 发布自检和逐文件校验 |
| 根目录 Markdown 文档、pyproject.toml、uv.lock | 方案、操作和依赖声明 |

除选定的 models/lamp_v7/best.pt 外，其他训练权重、训练视频、人工标注素材、tmp 备份、data/output、日志、私有 .env 和 config.local.yaml 不上传。
