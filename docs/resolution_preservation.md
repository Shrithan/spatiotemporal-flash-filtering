# Resolution-preserving processing design

The experimental YouTube outputs were downloaded or resized to 320 pixels
wide for rapid iteration. That was an experiment choice, not an algorithmic
requirement. `flashfilter` already writes the same dimensions it receives,
but full-resolution production processing needs a more efficient path.

## Proposed two-resolution architecture

1. Decode one full-resolution frame at a time as 8-bit RGB.
2. Downsample a copy to an analysis width such as 640 pixels.
3. Compute linear luminance, signed transitions, scene-cut evidence, and
   optical flow on the analysis copy.
4. Generate and feather the activity mask at analysis resolution.
5. Upscale the mask bilinearly to the exact source dimensions.
6. Apply reconstruction or luminance clamping to the full-resolution frame in
   overlapping tiles, limiting peak RAM use.
7. Pipe frames to FFmpeg using H.264/H.265 at a configurable CRF, then remux
   the original audio, timestamps, and selected metadata.

Only the detector runs at reduced resolution; the delivered pixels remain at
the source resolution. Mask coordinates scale proportionally, and feathering
is expressed relative to frame size rather than as a fixed pixel radius.

## Quality controls

- Assert output width, height, frame count, frame rate, and duration against
  the source with `ffprobe`.
- Use a visually lossless intermediate or direct FFmpeg pipe; avoid repeated
  MP4V encode/decode cycles.
- Report source-to-output VMAF or another codec-aware quality metric in
  addition to MAE/SSIM when tooling is available.
- Keep a lossless mask/video fixture to distinguish algorithmic distortion
  from codec distortion.
- Never evaluate a high-resolution source against a separately resized output.

## Frame-generation integration

Motion estimation can run on the analysis copy. Flow vectors are scaled to
full-resolution coordinates, and reconstruction is applied only within the
upscaled flash mask. A one-frame buffer permits reconstruction of frame B from
neighbors A and C without changing frame rate or duration.

## Remaining engineering work

- Implement the low-resolution detector/full-resolution renderer interface.
- Add tiled chromaticity-preserving luminance adjustment.
- Add FFmpeg pipe encoding and automatic audio remuxing to the CLI.
- Benchmark 720p, 1080p, and 4K memory, speed, and quality independently.
