# FNF Separator Android

Native Android prototype for the FNF vocal/instrumental separator.

## Current build

- Runs locally on Android 10+.
- Uses ONNX Runtime Android.
- Accepts Android-supported audio formats through the system file picker.
- Decodes/resamples to 44.1 kHz stereo.
- Runs the V3.1 separator in chunks with a progress bar.
- Writes `*_Vocals.wav` and `*_Instrumental.wav` to `Music/FNF Separator`.
- The first prototype imports `fnf_separator_v31.onnx` from storage. A later build will bundle the ONNX model directly in the APK assets.

## GitHub Actions

The workflow `.github/workflows/build-android.yml` builds a debug APK on pushes to `android-apk` and can also be started manually with `workflow_dispatch`.

The resulting artifact is named `FNF-Separator-Android-APK` and contains `FNF-Separator-v0.1-debug.apk`.

## Local build

```bash
cd android
gradle :app:assembleDebug
```

The APK is written to `app/build/outputs/apk/debug/app-debug.apk`.
