# BarriGuard YOLO

水马使用 V6 YOLO；灯具使用 YOLO 定位、同视角 ID 关联、完整闪烁计数。两者共用摄像头级偏移模块和参考图库。旧灯光算法及后端切换已移除，灯具跨视角旧 ID 迁移暂缓。

灯具固定 v7/320，v9c 已接入生产控制器，两路原始15分钟视频验证已完成。每周期15次可信完整闪烁即可正常；300有效秒用于零闪证据和全景搜索，未通过者最长观察25分钟，每30分钟重开。两份模板保持 night_lamp.enabled=false，服务器通过私有配置开启受控试运行后完成整夜和实时性能验收。详情见 [生产接线说明](PRODUCTION_WIRING.md)。

- [生产接线与灯具启用](PRODUCTION_WIRING.md)
- [完整方案](LAMP_DETECTION_PLAN.md)
- [当前实现状态](IMPLEMENTATION_STATUS.md)
- [完成版发布复核](RELEASE_REVIEW_20261102.md)
- [服务器部署](SERVER_DEPLOYMENT.md)
- [上传索引](UPLOAD_INDEX.md)
- [本地资料与备份](NOT_UPLOAD_INDEX.md)

水马参数入口为 deploy/app/water_barrier/yolo_params.py，权重为 models/water_barrier_v6/best.pt；灯具推理参数和观察规则位于配置中的 detector 与 observation。

服务器在项目根目录先执行 git pull --ff-only，再进入 deploy 构建与检查：

```bash
docker compose build --pull barriguard
docker compose run --rm --no-deps barriguard python -B check_deploy.py --load-model
docker compose up -d --no-build barriguard
```

必须先检查成功再更新容器。上传源码不会自动更新服务器或启用灯具；本机尚未验证Docker镜像构建及服务器整夜性能，没有启动实时识别或发送回调。本地视频验证使用项目tmp中的独立FFmpeg，不要求加入系统PATH。
