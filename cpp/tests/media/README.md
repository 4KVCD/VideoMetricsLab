# Test media

Small crafted streams the app tests read (`VML_TEST_MEDIA_DIR`), each made as below.

## vvc_gray_afgs1_chroma.266

335 bytes, SHA-256 `1663a86640f01c7180b16415f3d08113ca6c0f24a9ae589b3e7521f1d25ab54c`. One 64x64 8-bit
4:0:0 VVC picture (Annex B) after an AFGS1 film grain message (ITU-T T.35, AOM) whose one set has chroma
points and declares chroma subsampling 0,0, so that a monochrome picture selects it: FFmpeg 9.0.2's own VVC
decoder then writes chroma grain into planes the picture has not, and crashes, already in
`avformat_find_stream_info` (`app/native_probe_test.cpp`).

1. The picture: `ffmpeg -i <any 8-bit video> -frames:v 1 -vf crop=64:64:600:300 -pix_fmt gray -strict -1 tiny.y4m`,
   then VVenC 1.15.0-dev: `vvencFFapp -i tiny.y4m --preset faster --QP 40 --Threads 1 --InputChromaFormat 400
   --ChromaFormatIDC 400 --InternalBitDepth 8 -f 1 -b tiny400.266`.
2. The message: the AFGS1 message of an AFGS1 VVC stream made for VideoMetricsLab's tests (one set: 3 luma points,
   4 and 3 chroma points, not luma-only), with the set's width and height set to 0 (any picture), its bit depth to
   8 and its chroma subsampling to 0,0, as this prefix SEI NAL unit (type 23, payload type 4: an ITU-T T.35 message,
   country 0xB5, provider 0x5890, AOM's 0x01) put in front of the picture's slice, after its PPS:

   ```
   00 00 00 01 (start code), then
   00b90476b558900180388ed154000003000407d00a8db3c9c0981b000a29a28ba83e038000d9792bfdfe15d23a0dfa020dda4d9dee09fde6
   5962ca1df6065903b780817b89837f80807e9166798281808a53c1877d7f8561e077e01fdfa220e0206061601cdd2020201f58b024605fe0
   1d3520920302010181008080
   ```
