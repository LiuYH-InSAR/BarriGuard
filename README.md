# BarriGuard YOLO

水马使用V6 YOLO；灯具使用YOLO定位、持久ID关联与迁移、完整闪烁计数，以及有独立白天证据支持的实体缺失确认。两者共用摄像头级偏移模块和参考图库，水马和灯具均可反馈新视角实测位置参与校正。旧灯光算法及后端切换已移除。

灯具固定 v7/320，v9c 已接入生产控制器，两路原始15分钟视频验证已完成。每周期15次可信完整闪烁即可正常；300有效秒用于零闪证据和全景搜索，未通过者最长观察25分钟，每30分钟重开。生产配置 deploy/config.yaml 已启用两路灯具和回调，按20:00～03:00观察、07:00推送；config.test.yaml 保持灯具及回调关闭。服务器若挂载现有 config.local.yaml，仍需修改该私有文件中的开关并重新创建容器，pull不会覆盖它。详情见 [生产接线说明](PRODUCTION_WIRING.md)。

- [生产接线与灯具启用](PRODUCTION_WIRING.md)
- [完整方案](LAMP_DETECTION_PLAN.md)
- [服务器部署](SERVER_DEPLOYMENT.md)
- [上传索引](UPLOAD_INDEX.md)
- [本地资料与备份](NOT_UPLOAD_INDEX.md)

当前目录保留 `deploy/`（源码、配置和测试）、`models/`（V6和v7选定权重）、部署与功能说明、依赖和发布校验文件。训练数据、回放、标注、本地缓存及历史开发记录已完整归入 `tmp/development_cleanup_20261102_154735/`，不上传GitHub或进入Docker构建；新旧位置及最新结果图入口见本地资料索引。

水马参数入口为 deploy/app/water_barrier/yolo_params.py，权重为 models/water_barrier_v6/best.pt；灯具推理参数和观察规则位于配置中的 detector 与 observation。

服务器在项目根目录先执行 git pull --ff-only，再进入 deploy 构建与检查：

```bash
docker compose build --pull barriguard
docker compose run --rm --no-deps barriguard python -B check_deploy.py --load-model
docker compose up -d --no-build barriguard
```

必须先检查成功再更新容器。上传源码不会自动更新服务器或启用灯具；本机尚未验证Docker镜像构建及服务器整夜性能，没有启动实时识别或发送回调。本地视频验证使用项目tmp中的独立FFmpeg，不要求加入系统PATH。
