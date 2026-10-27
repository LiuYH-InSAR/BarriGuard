# GitHub与服务器部署

## 当前组成

水马为V6 YOLO缺失检测，灯光沿用BarriGuard的poly/circle标定及闪烁算法。只改水马后端和必要接入，不修改灯光识别逻辑。

模型在`models/water_barrier_v6/best.pt`；参数统一在`deploy/app/water_barrier/yolo_params.py`。默认CPU，调试总开关`Debug.enabled=true`。服务器不需要训练数据、旧模型或本地测试状态。

## 第一次服务器测试

先clone选定的GitHub仓库和分支。仓库地址及分支尚待确认，本目录未自动连接或推送。

```bash
cd <仓库目录>/deploy
cp config.test.yaml config.local.yaml
cp .env.example .env
```

按服务器情况检查`config.local.yaml`的视频流、回调地址、灯光录像路径。测试模板关闭callback，其他算法设置与生产配置相同。两份私有文件均不提交GitHub；后续pull不会覆盖。

```bash
docker compose build --pull barriguard
docker compose run --rm --no-deps barriguard python -B check_deploy.py --load-model
docker compose up -d --no-build barriguard
docker compose ps
docker compose logs --tail=100 barriguard
```

检查通过后再执行up；自检不启动HTTP、RTSP或callback。原服务使用相同project/service/container名称，更新会替换原实例。默认HTTP端口8000，可在`.env`修改宿主机端口。

如果旧服务器命令同时使用`docker-compose.blocks.yml`，兼容入口仍保留；实际水马后端统一为YOLO，不加载旧blocks配置。

## 后续更新

在服务器仓库根执行：

```bash
git pull --ff-only
cd deploy
docker compose build --pull barriguard
docker compose run --rm --no-deps barriguard python -B check_deploy.py --load-model
docker compose up -d --no-build barriguard
```

模型和代码应来自同一提交。检查失败先处理，旧容器可继续运行。不要用`git reset --hard`覆盖服务器改动；私有配置保留在`config.local.yaml`。

确认需要真实回调时，在服务器私有配置中打开`callback.enabled`并重启。没有`.env`时Compose默认使用`config.yaml`，其回调保持原来的开启状态。

## 镜像与数据

构建上下文为仓库根目录，保留`/app/deploy/app`拓扑，使水马的项目根路径为`/app`。模型在`/app/models/water_barrier_v6/best.pt`。[Docker构建上下文说明](https://docs.docker.com/build/concepts/context/)。

保留原BarriGuard的现代FFmpeg获取方式，用于这些摄像头的HEVC解码。补齐YOLO依赖，CPU Torch主版本与本机验证一致，来源为[PyTorch官方CPU索引](https://download.pytorch.org/whl/cpu/torch/)。GPU需另行核对服务器驱动和镜像参数。

| 宿主机 | 容器 | 内容 |
|---|---|---|
| `data/` | `/app/data` | 原服务器结果、日志、灯光录像及结果 |
| `output/water_yolo/` | `/app/output/water_yolo` | 水马中间结果、证据和检查点 |
| `output/model_settings/` | `/app/output/model_settings` | 推理库运行缓存 |

这些目录不提交GitHub，重建容器不会清除宿主机结果。服务器按自己的画面建立参照，不加载本地回放的累计和报警状态。

本机无Docker引擎，尚未实测Linux镜像构建和服务器性能。首次服务器部署须执行build及自检；本机规则测试不能代替真实缺失、长期停车和连续整夜验收。

## 水马中间结果

各摄像头分别在`output/water_yolo/<camera>/`保存：

| 子目录/文件 | 内容 | 条件 |
|---|---|---|
| `references/` | 外观图、质量、坐标变换 | 合格参照建立/更新 |
| `baselines/` | 正常排列、目标身份、推断空位 | 基准建立/更新 |
| `movement/` | 移动前后图及原因 | 确认摄像头移动 |
| `alignment_recovery/` | 对齐恢复证据、新外观 | 可靠恢复 |
| `reconstruction/` | 重构图与排列继承信息 | 重构完成 |
| `suspect/` | 候选区域图及原因 | 首次及持续期间按间隔更新，不报警 |
| `alarms/` | error图片、原因和统计 | 满足报警条件 |
| `observations/` | 分类、滑窗、累计 | 有效检测判断 |
| `debug/` | 对齐、画质、分块、匹配、状态、决策 | 仅调试总开关开启时 |
| `current.json` | 当前参照、累计、事实、模板、事件去重 | 原子更新，重启续跑 |

历史水马资产生成满72小时后按周期清理，停用参考不立即删除。`current.json`保存当前状态与持续事件去重，不按历史图片清理。日志不纳入水马72小时规则。

灯光仍沿用原策略：`data/night_offline/videos/`保留3天，`results/`保留7天；本次不改变灯光清理配置。

suspect本身不推送；符合资格的候选升级error后才报警。小目标、背景证据不足、弱存在和光照待确认不会直接累计成error。
