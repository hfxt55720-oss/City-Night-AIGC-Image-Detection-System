# GitHub 上传检查清单

上传前建议逐项确认：

- [ ] 没有提交 `build/`。
- [ ] 没有提交 `dist/`。
- [ ] 没有提交 `output/` 中的检测结果、热力图或训练输出。
- [ ] 没有提交 `datasets/` 中的训练或测试图片。
- [ ] 没有提交 `data/history.json`。
- [ ] 没有提交 `models/weights/*.pth`、`*.pt`、`*.onnx` 等权重文件。
- [ ] 没有提交软著申请资料、手册截图或个人登记信息。
- [ ] 已确认 `aide_external` 的第三方来源和许可证处理方式。
- [ ] 已在 README 或 Release 中说明权重下载方式。
- [ ] 已选择合适的开源许可证；如果暂不确定，先不要随意添加 LICENSE。

推荐上传内容：

```text
main.py
README.md
requirements.txt
.gitignore
config/
ui/
models/
inference/
training/
system/
tools/
data/img/
build_release.ps1
build_installer.ps1
installer_nsis.nsi
AIGC_Detector.spec
WEIGHTS.md
THIRD_PARTY_SETUP.md
```

