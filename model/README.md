# V3.1 model checkpoint

The Android GitHub Actions workflow looks for the PyTorch checkpoint at:

`model/best_model_v31.pt`

When that file is present, `.github/workflows/build-android.yml` automatically:

1. installs CPU PyTorch + ONNX dependencies;
2. runs `scripts/export_v31_onnx.py`;
3. creates `android/app/src/main/assets/fnf_separator_v31.onnx`;
4. compiles the APK with the ONNX model bundled inside;
5. uploads both the APK and generated ONNX as GitHub Actions artifacts.

If the checkpoint is absent, the APK still builds and keeps the manual ONNX import fallback.
