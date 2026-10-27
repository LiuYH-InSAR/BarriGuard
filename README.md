# BarriGuard YOLO

通过GitHub同步到服务器，再用Docker Compose构建、检查和更新运行。水马采用V6 YOLO；灯闪烁保留BarriGuard当前的poly标定及算法。

- [提交GitHub的文件](UPLOAD_INDEX.md)
- [不提交的本地资料及新位置](NOT_UPLOAD_INDEX.md)
- [服务器部署与中间结果说明](SERVER_DEPLOYMENT.md)
- 参数入口：`deploy/app/water_barrier/yolo_params.py`；调试总开关：`Debug.enabled`，当前开启。
- 模型：`models/water_barrier_v6/best.pt`，已放行Git忽略规则，不依赖训练目录。

服务器更新，在仓库根执行：

```bash
git pull --ff-only
cd deploy
docker compose build --pull barriguard
docker compose run --rm --no-deps barriguard python -B check_deploy.py --load-model
docker compose up -d --no-build barriguard
docker compose ps
```

先检查成功再替换容器。第一次测试请按部署说明创建私有配置，使用callback关闭的测试模板。原生产模板callback仍开启。

本机没有Docker引擎，不宣称已实测Linux镜像构建。已完成水马规则和录像验证；服务器仍需验证依赖下载、性能和现场行为。未连接或推送GitHub、未启动真实服务或发送回调，也没有制作压缩包。
