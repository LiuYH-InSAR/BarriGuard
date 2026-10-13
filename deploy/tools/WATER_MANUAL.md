# 固定水马单块模式

独立开关 `--manual-blocks`，不改变生产算法或原回放模式。去掉该开关回到旧模式。
标注需有 row0/row1 整排 polygon 及 row0block01 等单块 polygon，兼容下划线。
参考图必须是确认水马正常、与视频机位相同的画面。默认寻找标注 JSON 的 imagePath，
先查标注同目录，再查 tmp/roi_relabel；可用 --block-reference 指定。
旧视频与新标注机位不一致时，运行只用于检查输出格式，不能评价准确率。

```powershell
& "C:\Users\admin\venv-cu12\Scripts\python.exe" "C:\Users\admin\Desktop\projects\BarriGuard\deploy\tools\run_water_clip.py" --local-video --video "C:\Users\admin\Desktop\projects\BarriGuard\tmp\hourly10\1002490_20260918_1900.mp4" --camera 1749 --manual-blocks "C:\Users\admin\Desktop\projects\BarriGuard\deploy\app\roi_configs\blocks\1749.json" --diagnostics --matrix-every 1 --max-pages 20
```

默认检测前60秒，逐帧处理。--stop-after 修改测试长度。矩阵仅在 --diagnostics 开启时保存，
按 --matrix-every 秒、状态变化和 --at-s 指定时间保存；最多 --max-pages 份。
逐帧 block 汇总始终写入 blocks.jsonl；不保存每帧原图或每点小图。

每站按原2像素步长，在该站/block相交高度10%、30%、50%、70%、90%位置取5×5小块。
只有poly内像素参与统计。六特征：红比例、白比例（排除红重叠）、异色比例、V中位数、
S中位数、有效像素比例。红白HSV规则复用旧算法；整帧质量门复用旧算法。
点颜色类型从正常参考初始化，未知类型不强行判断；类型在运行中冻结。
同排邻块提供共同V变化和S辅助变化，参考不足、快速光变、曝光不可靠输出?。
当前曝光门采用中位数/整帧比例，并非完整的人车遮挡识别。
使用固定正常参考加邻块光照补偿，暂未自动更新参考，以免学习异常；跨昼夜需验证覆盖。

站至少3有效点、异常点比例≥0.6；块有效站比例≥0.7、至少5有效站、唯一采样像素≥100；
异常站比例>0.7产生*，有效异常持续10秒产生error；恢复5秒降为正常。
无法判断暂停增加证据；观察间隔>1秒时断开异常累计，不将空档计为持续异常。
奇偶只记录本体预期颜色，不强制上下点类型；不使用旧路面RER。
阈值集中在 ManualSpec，--manual-parameters 可传JSON覆盖，无需编辑主算法。
这些阈值尚未完成全天候准确率验证。

index.html：总结果、block表、各帧矩阵链接、每块时间曲线与状态带。
data/frame_*.npz：每块 features 为6×5n，anomaly 为6×n。
异常码0正常、1异常、-1无法判断；第6行为站判断。
geometry.json：站号对应五点图像坐标、点类型、唯一像素支持。
labels.json/reference.png/manifest.json：标定副本、正常参考、全部参数及哈希。
blocks.jsonl：每帧状态、覆盖率、异常比例、累计时间及光照诊断。
summary.png/json：首个最高异常状态的代表帧及其块汇总，不代表全程最后状态。

移除功能：删除 water_manual.py、test_water_manual.py、本说明，以及回放器的 manual
参数和分支即可。生产服务无此模块依赖。

## 可选改进（全部默认关闭）

| 开关 | 功能 |
|---|---|
| --gray-white | 加入灰白分支：V≥80、S≤70、RGB通道差≤40，排除红色重叠 |
| --light-colour | 邻块估计共同V变化后，对原采样像素的V归一化并重算颜色比例 |
| --abnormal-details | index中列出ERROR/疑似证据及对应时间 |
| --abnormal-only | 普通矩阵快照只保存异常块，并从index移除全块单帧区 |
| --matrix-labels | 特征浅→深蓝色标、状态图例、站号和P1–P5提示、参考及修正比例对照 |
| --independent-evidence | 每块首次疑似/首次error各保存一份证据，不消耗max-pages |
| --dual-scenes | 保存低运动场景图和原异常代表图，分别标时间和状态 |
| --validity-optimize | 独立有效性优化模块；部分窗口允许60%覆盖但至少15像素，最低站数随块站数调整，100个唯一像素底线不变 |

诊断相关开关配合 --diagnostics 使用；专项展示通常同时开启 --abnormal-details。
保留原6×5n特征矩阵，另外保存2×5n colour_evidence（实际用于判断的红/白比例）。
未开启光照修正时colour_evidence与原红白比例相同。未开启灰白分支时复用原HSV规则。
灰白分支仍可能把灰色背景当白色，这些阈值是待验证方案，不是完成准确率标定。
低运动场景选帧优先有效观察，再按相邻帧变化量择优；不能排除停驻车辆，不替换正常参考。
独立证据最多每块两份，文件名evidence_块号_suspected/error；普通页面额度不影响它。
有效性优化默认关闭；不加开关时不导入water_manual_validity.py，保留原80%点覆盖/5站规则。
开启时输出reference_validity.json及网页参考可检测性表，区分参考不可检测与本帧观察不足。
small_point_valid、small_point_pixels、small_min_stations可通过--manual-parameters调整。
关闭--diagnostics只关闭矩阵和证据导出，blocks.jsonl及最终网页仍保存。
ManualDetector.step本身不写文件，可供在线调用；目前生产服务仍未接入此独立模式。


### 可选小范围优化（2026-10-12）

在 `--colour-regions` 基础上增加：
- `--region-support`：色块不足，但当前或参考至少2站有弱同色证据（比例≥colour_min且同色占优）时，输出无法判断，避免直接确认缺失。共同光照与曝光检查沿用原路径。这是保守证据门槛，不能完整证明颜色可识别性。
- `--region-bridge`：按实际坐标和原V/S相似条件，最多跨过1个站连接候选点。跳过的点不会增加站数或像素支持。

两个开关默认关闭，参数JSON对应 `region_support`、`region_bridge`；关闭需同时去掉CLI参数并将JSON字段设false。
门槛 `region_weak_stations` 默认为2，可用参数JSON调整。支持详情保存于blocks.jsonl和网页regions.colour_support。
采样、颜色阈值、历史累计规则不变；无法判断仍沿用原累计暂停规则，不代表消除整段误报。
本次修改前备份：`tmp/backups/water_before_region_refine_20261012_170656.zip`，包含原相对目录，恢复至项目根目录。


### 弱颜色与近期累计（2026-10-12）

在原色块检测命令末尾增加 `--weak-red --weak-white --recent-missing`；均默认关闭。
CLI关闭还需保证参数JSON对应 weak_red/weak_white/recent_missing=false。

- 弱红：RGB中R至少为G/B的1.15倍，且至少高10；V不低于dark_v。
- 弱白：RGB最大最小差/最大值≤0.35，点S≤90，V不低于dark_v。灰背景可能符合，只作保守无法判断证据。
- 同色比例至少0.2，复用真实坐标连通、V/S相似、跨站数、唯一像素和覆盖检查。弱证据只用于强色块不存在时阻止缺失确认，不直接判正常。
- 近期累计：只统计最近hold_s（默认10秒）内两端均为缺失的时间段。窗口需已满，支持达到95%才确认；UNKNOWN不投缺失票，正常会清空，长采样中断也清空。短UNKNOWN可容忍但不能无限保留旧累计。累计显示的是窗口内缺失秒数，所以9.5秒以上可能ERROR。

阈值JSON字段：weak_red_ratio=1.15，weak_red_delta=10，weak_white_spread=0.35，weak_white_s=90，weak_colour_min=0.2，recent_missing_fraction=0.95。
弱颜色详情保存在regions.weak_colour（含候选站、连通区域、支持门槛），可在专项诊断页面与blocks.jsonl查看。既有6×5n特征矩阵不变。
修改前备份：tmp/backups/water_before_weak_recent_20261012_172156.zip（原相对目录）。

验证：8项测试通过；实际首帧07疑似、13正常、15/17弱红证据不足确认而无法判断。对上一轮750帧证据做近期累计回放，07约10.01秒确认，15没有确认；这不等于新版本完整视频验证。


### 可选检测抽帧（2026-10-13）

`--detect-hz 2`：每视频秒约检测2帧；默认0逐帧。仅手动block本地回放支持。
保留源视频FPS与frame/FPS时间戳，解码跳过的帧不运行检测；原poly/颜色/累计算法不变。
时间选择采用首个达到目标时刻的源帧，非整数帧率可能稍有偏差。采样中断门槛至少为1/频率+1/源FPS+微小容差。
低频观察不能证明两次采样间持续缺失；UNKNOWN间隔仍不投缺失票。
若同时输出overlay视频，跳过的帧复用最近检测状态，保持源FPS与时长。诊断择帧仅在检测帧进行。
manifest新增decoded_frames（源帧）、detected_frames（检测次数）、detect_hz、elapsed_before_report_s、elapsed_total_s。
网页显示报告生成前耗时；终端与manifest显示总耗时（含初始化、检测、导出，不含最后一次manifest写入）。
关闭：移除--detect-hz或设置0。9项测试通过，12秒20FPS样例240帧抽为24次检测，缺失在第10视频秒确认。
备份：tmp/backups/water_before_sampling_20261013_082116.zip。


### 小block补检通道（2026-10-13）

`--small-fallback` 默认关闭。仅主通道单帧evidence=UNKNOWN、站数≤12时进入。
复用当前采样和修正后颜色比例；按真实坐标去除重叠点，不强求每站3个点。
每站至少25个唯一有效像素；有效站覆盖≥70%，至少2个站；全块仍需100个唯一像素。
强同色连通区域仍要求原40%点比例、20%跨站（至少2站）、50个唯一像素。
曝光/整帧质量/同色光照参考合格，且没有原异常站冲突时，仅补判正常；不足仍UNKNOWN，不从补检输出ERROR。
poly、原特征/异常矩阵保留。small_fallback详情记录实际像素、覆盖、质量、冲突和recovered结果；网页详情及blocks.jsonl可查看。
fallback_max_stations=12、fallback_station_pixels=25可在参数JSON调整；关闭需CLI删除且JSON small_fallback=false。
10项测试通过。上一轮总图对应帧补判row0block22/34/58、row1block10正常；整视频效果未验证。
备份：tmp/backups/water_before_small_fallback_20261013_082346.zip。


### 整块像素补检（2026-10-13）

`--pixel-fallback` 默认关闭，可与小block补检共存：主通道及小block通道仍UNKNOWN才进入。
候选：站数≤fallback_max_stations（默认12），或唯一采样面积/高度≤patch_size（默认5）。row1block01不进入。
直接读取poly内部全部像素，复用原HSV红/白标准、灰白开关、同色V补偿及质量门槛。
存在门槛独立于主通道100像素：pixel_min_pixels=30；有效像素/poly≥70%；同色/有效像素≥20%；最大8邻域同色区域≥max(pixel_component_min=12，有效像素×pixel_component_fraction=0.15)。
当前同色比例相对参考下降≥colour_loss=0.25仍UNKNOWN；不因单个原异常站否决。通过只补判正常，未通过保持UNKNOWN，不输出缺失票。
灰白背景仍可能符合白色存在证据，需实测缺口对照；此通道不提供独立物体身份识别。
诊断pixel_fallback含poly/有效/同色像素、最大区域及门槛、参考损失、质量、recovered，见blocks.jsonl与页面详情。原矩阵不变。
关闭：去掉CLI开关并确保参数JSON pixel_fallback=false。11项测试通过，专项单帧恢复18块；不是完整视频验证。
备份：tmp/backups/water_before_pixel_fallback_20261013_083220.zip。


### 整块补检默认开启／简化命令（2026-10-13）

本地manual回放现在默认开启pixel_fallback，不再需要--pixel-fallback。
关闭用--no-pixel-fallback；参数JSON pixel_fallback=false也可关闭；显式CLI开关优先于JSON。
此默认仅用于回放入口；ManualSpec直接实例化的默认不变。其他功能默认不变。
water_manual_test.json收纳当前测试使用的10个算法开关，可通过--manual-parameters加载，代替这10个CLI开关。导出、抽帧、视频/poly参数仍在CLI。
备份：tmp/backups/water_before_default_pixel_20261013_083514.zip。11项测试通过，含默认开启与显式关闭。


### 总图跳过开头（2026-10-13）

总图默认跳过前5视频秒（--summary-skip 5）。只限制summary择帧，不跳过检测、累计或矩阵。
--summary-skip 0恢复原行为；候选仍按ERROR/疑似数量排序。
短视频若没有5秒后的检测帧，回退最后检测帧并在网页和manifest注明。
manifest包含summary_skip_s和summary_short_video_fallback。11项测试通过。
备份：tmp/backups/water_before_summary_skip_20261013_091419.zip。

### 同帧相对红色与缺失证据门槛（2026-10-13）

两个独立开关relative_red、missing_support，ManualSpec默认关闭；water_manual_test.json启用。
关闭任一功能：对应JSON值设false，并删除对应CLI --relative-red或--missing-support。
只作用于奇数红色block；白色、原光照邻块搜索、原累计逻辑不变。

relative_red：找同排最近6个其他奇数block，排除上次疑似/ERROR。
校准像素必须来自参考已标为红色且当前原红比例>=20%的有效采样点。
至少2个邻块各有25个唯一有效红像素才使用；红色H距0/180<=20，V>=25。
当前阈值S>=max(15,邻块红像素S中位数*0.2)，归一化RGB红优势>=max(0.03,邻块中位数*0.2)。
比例加入colour_evidence红色通道，仍按原连续区域面积、跨站、质量及光照门槛判存在。
原6行features及异常矩阵不改；旧弱颜色保护使用新增比例前的输入，避免零散红点扩大保护。

missing_support：主通道拟投缺失票时检查校准可靠、实际有效点/站覆盖，以及放宽红色证据。
放宽红色仅使用S>=15、RGB优势>=0.03及上述H/V限制。
有效站至少3个有效点；其中至少60%点无放宽红色，才计缺失支持站。
块有效站覆盖>=70%、至少5站、缺失支持站占比>70%，才保留缺失票；不足改UNKNOWN不累计缺失。
这是颜色缺失支持门槛，不能单独鉴别人车遮挡。原区域质量与面积检查仍先执行。
blocks.jsonl/详情记录relative_red阈值、邻块数、候选点数和missing_support覆盖/缺失支持比例。

备份：tmp/backups/water_before_relative_red_20261013_102431.zip。
