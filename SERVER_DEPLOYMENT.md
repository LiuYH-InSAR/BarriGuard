# GitHub与服务器部署

## 当前组成

水马为V6 YOLO缺失检测；灯具使用固定v7权重定位、完整闪烁计数和独立业务窗口。两者共用偏移模块和参考图库，旧灯光poly/circle、热力图及周期性后端已移除。灯具跨视角旧ID迁移暂缓，不把预测位置当作已验证身份。

模型在`models/water_barrier_v6/best.pt`；参数统一在`deploy/app/water_barrier/yolo_params.py`。默认CPU，调试总开关`Debug.enabled=true`。服务器不需要训练数据、旧模型或本地测试状态。

## 第一次服务器测试

仓库为 `https://github.com/LiuYH-InSAR/BarriGuard.git`，发布分支为 `work/current-water-lamp`（本地 `main` 跟踪该分支）。服务器应clone该分支或切换到该分支后更新，不使用旧 `master`：

```bash
git clone --branch work/current-water-lamp https://github.com/LiuYH-InSAR/BarriGuard.git
```

```bash
cd <仓库目录>/deploy
cp config.test.yaml config.local.yaml
cp .env.example .env
```

按服务器情况检查`config.local.yaml`的视频流、回调地址和输出目录。测试模板关闭callback，灯具仍禁用；在服务器试运行时将所需摄像头的 `night_lamp.enabled` 改为 `true`，保留v7权重路径、320输入和观察参数。两份私有文件均不提交GitHub；后续pull不会覆盖。

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

灯具已统一为 v9c 新方案，已有本地候选权重；配置在验收前保持禁用。启用、权重挂载和 output/lamp_state 七天历史清理见 PRODUCTION_WIRING.md。

suspect本身不推送；符合资格的候选升级error后才报警。小目标、背景证据不足、弱存在和光照待确认不会直接累计成error。

灯具启用：仓库模板保持 `enabled=false`；服务器受控试运行通过私有 `config.local.yaml` 开启，开始时保持callback关闭。镜像已包含 `models/lamp_v7/best.pt`，无需额外挂载；仅需替换该权重时，设置 `BARRIGUARD_LAMP_WEIGHTS_FILE` 后使用可选的 `docker-compose.lamp.yml`。公共参考状态位于 `output/camera_motion`，灯具状态位于 `output/lamp_state`，默认compose已持久化。

正常为同一周期15次可信完整闪烁，无300秒的正向门槛；零闪证据至少300有效秒。未通过者继续到25分钟截止，每30分钟重开。03:00停止采样，07:00统一出图，状态按绿/红/灰显示。灰色是观察或身份不足，不能据此判坏。发布前合成时钟测试覆盖跨夜、恢复、空窗口及调度边界；真实整夜、服务器双路性能与网络解码仍需试运行验收。
