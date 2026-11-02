# 上传索引

服务器通过 Git 获取源码，再构建容器。

当前上传目标为`origin/work/current-water-lamp`（本地`main`的既有上游），保留旧`master`。上传前整理目录、验证、备份，再提交推送；上传不自动启用服务器灯具。

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
| README、LAMP_DETECTION_PLAN、PRODUCTION_WIRING、SERVER_DEPLOYMENT、上传／本地索引 | 当前方案、部署操作及目录入口 |
| pyproject.toml、uv.lock | 依赖声明与锁定 |

除选定的 models/lamp_v7/best.pt 外，其他训练权重、训练视频、人工标注素材、tmp 备份、data/output、日志、私有 .env 和 config.local.yaml 不上传。

整理后的训练、标注、回放、缓存和开发记录实际位于`tmp/development_cleanup_20261102_154735/`；旧备份、原始录像及此前归档留在原tmp子目录，不合并、不覆盖。历史`IMPLEMENTATION_STATUS.md`与`RELEASE_REVIEW_20261102.md`退出根目录和当前上传清单，归入该目录的`development_notes/`。回归测试属于当前代码验证，保留`deploy/tests/`。
