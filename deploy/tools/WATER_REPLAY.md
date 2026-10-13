# 水马本地视频与过程检查

生产服务不导入这些离线工具。本地回放默认关闭，必须指定 `--local-video`；
过程导出默认关闭，必须另加 `--diagnostics`。不调用平台回调，不受白天时段限制。

## 本地视频（独立功能板块）

PowerShell，使用项目已有虚拟环境。将视频路径替换成实际文件：

```powershell
& C:\Users\admin\Desktop\projects\BarriGuard\.venv\Scripts\python.exe C:\Users\admin\Desktop\projects\BarriGuard\deploy\tools\run_water_clip.py --local-video --video "D:\videos\1749_day.mp4" --camera 1749 --overlay-video
```

`--camera 1750` 使用1750标定。视频画面必须与标定机位匹配；当前路面采样框使用绝对像素，
生产标定来自1920×1080画面，其他尺寸需提供匹配标定的配置文件。

时间使用从0开始的 `frame_index / FPS`，不使用处理耗时。逐帧读取，不跳帧、不循环。
适用于固定帧率视频；FPS缺失或需明确覆盖时使用 `--fps 25`。
默认只检查前60秒，适合功能测试。`--stop-after 15` 改为前15秒；
只有显式指定 `--full-video` 才检查全视频。仍从开头累计历史。

每次运行都输出`summary.html`、`summary.png`和`summary.json`，无需开启diagnostics。
总结果图使用报警框最多、其次疑似框最多的一张实际帧；红框为算法报警，橙框为疑似。
标定poly范围用20%不透明度的浅绿色填充并标出区域编号；先画范围再画异常框。
汇总表记录整个片段中每条轨道的最高状态及异常首次/末次时间，轨道数不等于真实缺口数。
终端输出检查时长、报警轨道数、仅疑似轨道数和总结果路径；diagnostics总览也展示总结果图。

当前颜色分数为`0.6×(下部红色比例+下部白色比例)+0.4×上部白色比例`。
下部同一像素只计一次，白色比例扣除与红色掩膜重叠的像素；上60%/下40%分区保持不变。
异色规则独立保持不变，阴影白色仍可能同时触发异色识别。

## 窗口分块预览（独立模块和开关）

在原运行命令后加`--block-windows`，即可在最终`summary.png`中画蓝色窗口框。
不加此参数时不导入、不执行`water_blocks.py`。模块仅分析总结果的代表帧，
不改变缺口检测、历史累计、轨道状态和报警结果，也不把蓝框作为真实水马数量。

处理顺序：分别提取下部红、白比例→分别使用原高斯平滑→判断优势颜色→连续同色站合为一段。
优势颜色比例须达到各自阈值；两色相等、均低于阈值或无效站会断开区间。
index总结果下方展示独立的`blocks.png`，不叠加报警框。
蓝框R/W分别表示红/白优势段，站范围为0–811，像素框复用原坐标映射。
每段的颜色、站范围、宽度、平均优势比例和有效比例在总结果页查看，数组保存在`summary.json`。

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--block-score-min` | 0.35 | 平滑后红色优势比例下限 |
| `--block-white-score-min` | 0.05 | 平滑后白色优势比例下限，独立于红色 |
| `--block-width-ratio` | 0.60 | 局部参考宽度/poly高度，仅保存供对比，不强制切段 |
| `--block-valid-ratio` | 0.90 | 区间有效站占比下限；当前无效站直接断开，保留段全部有效 |

蓝框可能拆分或合并实际水马；本阶段只用于验证。遮挡、低光照区域可能无蓝框。
`summary_raw.png`保留无标注代表帧，以便后续重画不必重新跑视频。

关闭：去掉`--block-windows`。删除：移除`water_blocks.py`及回放器、总结果中的
对应可选调用块和CLI参数；生产算法无此模块依赖。

## 中间过程检查（独立开关）

```powershell
& C:\Users\admin\Desktop\projects\BarriGuard\.venv\Scripts\python.exe C:\Users\admin\Desktop\projects\BarriGuard\deploy\tools\run_water_clip.py --local-video --video "D:\videos\1749_day.mp4" --camera 1749 --diagnostics --overlay-video --page-every 30 --at-s 12 --at-s 45
```

默认输出到 `BarriGuard/output/water_clip/视频名_相机_运行时间/`，也可用 `--out` 指定空目录。
已有非空目录会拒绝覆盖。完成后打开输出目录的 `index.html`。

- `manifest.json`：运行设置、全部判断参数、标定、源代码哈希、完成状态与快照选择。
- `frames.jsonl` + `data/`：每帧真实判断记录和数组，包括淘汰候选、support、轨道、RER检查。
- `pages/` + `images/`：按代码顺序的单帧过程页、原图、掩膜、采样分区和中值图。
- `track_*.html` + `rer_*.html`：全程/开头/中间/结尾/状态变化曲线，以及实际RER确认和逐patch判断。
- `overlay.mp4`：仅在指定 `--overlay-video` 时生成。

详细快照在候选/质量门/轨道状态变化、定时检查和指定时间生成。
`--max-pages 20` 是默认详细快照上限；每个快照展示全部行。达到上限后仍保存全部逐帧trace与实际RER确认。
长视频的逐帧数组和实际确认图会占用磁盘，应先用 `--stop-after` 跑短片检查。
曲线保留全部样本，缺失值断线，阈值标线，失败项红字，未执行项为SKIP。
RER点只代表实际重算；缓存值单独显示。

主过程页按红白比例、颜色分数、红色/白色/异色识别、整帧质量、有效区域、
历史累计和路面判断展示图示，参数直接标在对应图中。`plots/`保存SVG及PNG。
异常区间全部标记，正常值按显示间距疏排；路面异常块全部标红，逐块数值在可展开的异常标注图中。
旧判断表保留在“补充详情”，不占主页面。

schema=2开始，选中快照保存独立原始HSV和红/白/异色掩膜，保留白色与异色同时命中的情况。
旧结果的颜色像素图只能从JPEG重建，页面明确标为示意；其他已保存判断仍使用原始trace。

## 只重画已保存数据

```powershell
& C:\Users\admin\Desktop\projects\BarriGuard\.venv\Scripts\python.exe C:\Users\admin\Desktop\projects\BarriGuard\deploy\tools\run_water_clip.py --replot "C:\Users\admin\Desktop\projects\BarriGuard\output\water_clip\某次运行"
```

只读取本次保存的参数、图像和数组，不重新打开视频，不用当前配置替换旧阈值。
重画默认只读前60秒，最多20张已保存快照，超出数量时沿时间均匀选取。
`--max-pages 3`可先预览3张；`--full-video`可重画完整记录。
详细原图只保存在已选快照处；其他帧保留信号和判断数据，不能在重画时凭空补出原图。

## 关闭或删除

不加 `--diagnostics` 即关闭中间过程导出；不运行本工具即关闭本地回放。
删除 `run_water_clip.py`、`water_intermediates.py`、`water_plot.py`、`water_result.py`、`water_blocks.py`、本说明和对应测试即可移除离线工具。
算法里的可选 `trace=None` 参数默认为关闭，无需为删除工具改生产服务。

除下部改为红白计分外，其余检测逻辑保持原样：support全历史累计；质量跳过可保留旧候选并继续上层计时；
RER降级计数按更新调用推进。报告记录这些行为，不修正算法。

## 验证

```powershell
& C:\Users\admin\Desktop\projects\BarriGuard\.venv\Scripts\python.exe -m unittest discover -s C:\Users\admin\Desktop\projects\BarriGuard\deploy\tests -p test_water_intermediates.py -v
```
