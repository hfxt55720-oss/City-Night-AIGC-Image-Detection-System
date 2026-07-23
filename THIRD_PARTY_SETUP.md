# 第三方 AIDE 依赖准备

本项目的 AIDE-Night 推理和训练功能需要 AIDE 相关外部依赖源码。公开 GitHub 发布版默认不附带这部分源码，请用户根据原始 AIDE 项目的许可证和使用要求自行准备。

放置结构如下：

```text
aide_external/
  models/
    AIDE.py
    srm_filter_kernel.py
    utils.py
  data/
    __init__.py
    datasets.py
    dct.py
```

准备完成后，再将 AIDE 权重放入：

```text
models/weights/aide_night_best.pth
```

如果不使用 AIDE-Night，只使用 ResNet50、EfficientNet-B0 或自定义模型，可以暂不准备 `aide_external`。

