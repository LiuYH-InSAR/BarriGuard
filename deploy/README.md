# 运行部署

使用项目根目录的 SERVER_DEPLOYMENT.md 与 PRODUCTION_WIRING.md。灯具仅有新 YOLO 方案，原 poly、热力图与周期性后端已移除。

当前 config.yaml 与 config.test.yaml 已固定 models/lamp_v7/best.pt、输入320，v9c 已接入生产控制器；验收前保持灯具实时禁用，水马仍可运行。有效窗口及实时性能验收通过后再设置 enabled=true。不要添加旧 backend、lamp_checks 或 periodicity 配置。
