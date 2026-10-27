# 提交GitHub的索引

服务器拉取这些文件，再构建运行，不使用上传压缩包。

| 路径 | 用途 |
|---|---|
| `deploy/app/server/` | API、视频流、调度、回调和清理 |
| `deploy/app/water_barrier/` | V6水马后端、参数及兼容导入模块 |
| `deploy/app/night_lamp/` | 保留的BarriGuard灯闪烁算法及内部依赖 |
| `deploy/app/roi_configs/1749.yaml`、`1750.yaml` | 灯光仍引用的几何范围，不能因水马改YOLO而移除 |
| `deploy/app/roi_configs/lamps/1749.json`、`1750.json` | 当前灯具poly/circle标定 |
| `deploy/config.yaml`、`config.test.yaml` | 生产模板、关闭callback的测试模板 |
| `deploy/Dockerfile`、`docker-compose.yml` | GitHub更新后构建和运行 |
| `deploy/docker-compose.blocks.yml` | 旧启动命令兼容入口，实际使用YOLO |
| `deploy/requirements-server.txt`、`.env.example` | 容器依赖及私有配置示例 |
| `models/water_barrier_v6/best.pt`、`model_info.json` | 唯一运行模型及SHA256 |
| `pyproject.toml`、`uv.lock`、`.python-version` | 保留原uv声明和锁定文件 |
| `.gitignore`、`.gitattributes`、`.dockerignore` | 排除本地资料、控制换行和构建上下文 |
| `check_deploy.py`、`deploy_manifest.json` | 启动前检查及逐文件校验清单 |
| `README.md`、`SERVER_DEPLOYMENT.md`、两份INDEX | 操作、结果位置和归档索引 |

V6约19MB，已在`.gitignore`单独放行，不要沿用旧项目“一律忽略*.pt”而漏传。当前不依赖Git LFS。

完整路径和校验值见`deploy_manifest.json`。配置和调参文件允许自定义；代码或模型完整性错误应先处理再更新容器。

未存储GitHub凭证，未自动连接或推送，也未启动服务器或发送回调。
