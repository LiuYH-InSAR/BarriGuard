逐block线上适配
================

启用：用config_blocks.yaml启动server。原config.yaml保持legacy，可随时切回。
逐摄像头water_gap.backend可取legacy/blocks；blocks必须提供manual_blocks、
manual_reference、manual_parameters和正数detect_hz（当前2）。
原calibration保留以便切回；legacy的detect/track/confidence不用于blocks判定。

直接复用deploy/tools/water_manual.py的ManualDetector和辅助模块，不调用replay、
网页、矩阵或plot导出。部署必须包含deploy/tools源文件。
线上参数独立保存在tools/water_manual_online.json，避免修改测试参数影响线上。
近期累计强制开启，hold_s取water_gap.alarm_hold_s（10秒）。
参考固定副本：app/roi_configs/blocks/references/{1749,1750}.png。
1749来自已标注原图；1750来自1750_20261013_100240.png。
沿用原day时段及夜灯配置，未新增水马全天或自动参考更新。

正常/疑似/UNKNOWN不发缺失告警；确认ERROR后生成现有water_gap alarm事件。
payload包含block_id、row_id、block_number、episode_id、box、accum_s及证据时间。
confidence为缺失支持时间/确认窗口的比例，不是模型概率。
每block独立事件；持续缺失按alarm_min_interval_s（3600秒）提醒。
UNKNOWN不重开事件；实际NORMAL连续reset_s（5秒）后结束事件。
恢复后再次确认ERROR生成新episode，Reporter允许绕过上一事件的间隔。
不推单独恢复通知；进程重启重建检测状态，重新确认后生成新事件。

只在内存保留最近5个检测帧（2Hz约2秒范围），告警图片候选必须含该块缺失票，
优先选变化较小、有效覆盖较高的一帧。目标块画红框，row浅绿20%透明。
没有更好候选则使用实际缺失证据；事件携带JPEG，worker不能用当前帧覆盖。
这些是异常证据帧，不使用后来正常的干净帧假装缺失证据。
无车辆身份识别或plot mask排除功能；原有无法判断检查仍生效。

仍复用Reporter的callback、重试、队列和告警证据落盘；告警落盘不是调试导出。
原小时快照机制保持：本小时有alarm才在到期时推最近图，并非缺失确认票。
测试不调用实际callback；上线命令会使用config_blocks.yaml中的既有callback地址。

本地：在deploy/app目录执行项目.venv的python -m server --config ../config_blocks.yaml。
Docker：在deploy目录执行docker compose -f docker-compose.yml -f docker-compose.blocks.yml up -d --build。
Dockerfile额外打包water_manual模块及线上参数到/tools，原默认配置仍为legacy。

回溯：备份tmp/backups/water_before_online_blocks_20261013_142619.zip，
其目录结构可解压回项目根。新增文件可保留不用，启动原config.yaml即切回原算法。
