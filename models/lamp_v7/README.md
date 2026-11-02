# Lamp v7

Runtime weight: `best.pt`, YOLO11s, input size **320**. The expected weight hash
and training reference are in `model.json`. Both deployment templates point to
this model; live lamp processing remains disabled pending video acceptance.

This is the selected detection baseline. It does not certify the complete
tracking, occlusion or flashing-health pipeline. Do not substitute input size
640/960 without a separate evaluated experiment.
