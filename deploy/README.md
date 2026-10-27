# 服务器更新入口

完整步骤见仓库根[部署说明](../SERVER_DEPLOYMENT.md)。

服务器在仓库根`git pull --ff-only`后进入本目录：

```bash
docker compose build --pull barriguard
docker compose run --rm --no-deps barriguard python -B check_deploy.py --load-model
docker compose up -d --no-build barriguard
docker compose ps
```

检查成功再替换容器。使用`.env`选择`config.local.yaml`，避免Git更新覆盖服务器私有配置。初次测试从`config.test.yaml`复制，callback关闭。

保留原FFmpeg、灯光算法和数据挂载；新增YOLO依赖、V6模型和水马结果持久化。默认CPU。本机尚未实测Docker构建，首次服务器构建需验证依赖下载与性能。
