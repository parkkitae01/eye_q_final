// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : 메인 앱 (카메라 → YOLO → 트래커 → HUD)
// ════════════════════════════════════════════════════════════════
// ⚠️ 수정 이력
//  1) YUV→RGB 변환을 compute()로 별도 isolate에서 실행 (UI 스레드 블로킹 제거)
//  2) FPS 누적 통계 + 워밍업 프레임 제외 + 구간별 소요시간(conv/detect/track)
//  3) [이번 수정] detect 내부 구간(lbox/fill/send/run/recv/flat/post)까지 HUD 표시
//     + FPS 배지가 상태바 시계와 겹치던 문제 수정 (상태바 높이만큼 HUD를 아래로 내림)
// 기존 lib/main.dart 내용을 전부 지우고 이걸로 교체하세요.
// ════════════════════════════════════════════════════════════════

import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart'; // compute(), debugPrint
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import 'object_tracker.dart';
import 'risk_engine.dart';
import 'yolo_detector.dart';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const EyeQApp());
}

class EyeQApp extends StatelessWidget {
  const EyeQApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EYE-Q',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const DetectionScreen(),
    );
  }
}

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  CameraController? _camera;
  final YoloDetector _detector = YoloDetector();
  final ObjectTracker _tracker = ObjectTracker();

  List<Detection> _detections = [];
  Detection? _top;
  Size _imageSize = Size.zero; // 카메라 프레임 해상도 (좌표 변환용)
  double _fps = 0;
  bool _busy = false; // 추론 중이면 프레임 스킵 (밀림 방지)
  int _lastMs = DateTime.now().millisecondsSinceEpoch;

  // ── 성능 측정용 ──
  static const int _warmupFrames = 10; // 초기 N프레임은 통계에서 제외
  int _frameCount = 0; // 처리한 전체 프레임 수
  int _statFrames = 0; // 통계에 포함된 프레임 수 (워밍업 이후)
  int _statStartMs = 0; // 통계 시작 시각
  double _fpsMin = double.infinity;
  double _fpsMax = 0;
  double _fpsAvg = 0; // 통계 구간 평균 FPS = 프레임 수 / 경과 시간
  double _sumConvMs = 0;
  double _sumDetectMs = 0;
  double _sumTrackMs = 0;
  // detect 내부 구간 누적: [letterbox, fill, send, run, recv, flatten, post]
  final List<double> _sumSub = List<double>.filled(7, 0);

  @override
  void initState() {
    super.initState();
    _setup();
  }

  Future<void> _setup() async {
    await _detector.init();
    _camera = CameraController(
      _cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await _camera!.initialize();
    if (!mounted) return;
    await _camera!.startImageStream(_onFrame);
    setState(() {});
  }

  Future<void> _onFrame(CameraImage image) async {
    if (_busy || !_detector.isReady) return;
    _busy = true;

    final now = DateTime.now().millisecondsSinceEpoch;
    final dt = (now - _lastMs) / 1000.0;
    _lastMs = now;

    try {
      final yPlane = image.planes[0];
      final uPlane = image.planes[1];
      final vPlane = image.planes[2];

      // ① YUV→RGB 변환 (별도 isolate)
      final swConv = Stopwatch()..start();
      final rgb = await compute(
        _convertYuv420,
        _YuvFrame(
          yBytes: yPlane.bytes,
          uBytes: uPlane.bytes,
          vBytes: vPlane.bytes,
          width: image.width,
          height: image.height,
          yRowStride: yPlane.bytesPerRow,
          uvRowStride: uPlane.bytesPerRow,
          uvPixelStride: uPlane.bytesPerPixel ?? 1,
        ),
      );
      swConv.stop();

      // ② 탐지 (전처리 + ONNX 추론 + 후처리 전체)
      final swDet = Stopwatch()..start();
      // 카메라 프레임은 센서 방향(가로)으로 오므로, 세워서 탐지 + 세워진 크기 기준으로 좌표 사용
      final int rot = _camera!.description.sensorOrientation;
      final raws = await _detector.detect(rgb, rotation: rot);
      final bool swap = rot == 90 || rot == 270;
      final double iw = (swap ? rgb.height : rgb.width).toDouble();
      final double ih = (swap ? rgb.width : rgb.height).toDouble();
      swDet.stop();

      // ③ 트래킹 + 위험도 계산
      final swTrk = Stopwatch()..start();
      final dets = _tracker.update(
        raws,
        dt,
        iw,
        ih,
      );
      final top = _tracker.topThreat(dets);
      swTrk.stop();

      final double fpsNow = dt > 0 ? 1 / dt : 0;

      // ── 통계 갱신 (워밍업 이후 프레임만) ──
      _frameCount++;
      if (_frameCount == _warmupFrames) {
        _statStartMs = now; // 워밍업 마지막 프레임 시점부터 통계 시작
      } else if (_frameCount > _warmupFrames) {
        _statFrames++;
        if (fpsNow > 0) {
          if (fpsNow < _fpsMin) _fpsMin = fpsNow;
          if (fpsNow > _fpsMax) _fpsMax = fpsNow;
        }
        _sumConvMs += swConv.elapsedMicroseconds / 1000.0;
        _sumDetectMs += swDet.elapsedMicroseconds / 1000.0;
        _sumTrackMs += swTrk.elapsedMicroseconds / 1000.0;

        final tm = _detector.timings;
        _sumSub[0] += tm.letterbox;
        _sumSub[1] += tm.tensorFill;
        _sumSub[2] += tm.send;
        _sumSub[3] += tm.run;
        _sumSub[4] += tm.receive;
        _sumSub[5] += tm.flatten;
        _sumSub[6] += tm.post;

        final double elapsedSec = (now - _statStartMs) / 1000.0;
        _fpsAvg = elapsedSec > 0 ? _statFrames / elapsedSec : 0;

        // 50프레임마다 콘솔에도 기록 (복사해서 보고서에 쓰기 편하게)
        if (_statFrames % 50 == 0) {
          String s(int i) => (_sumSub[i] / _statFrames).toStringAsFixed(1);
          debugPrint(
            '[EYE-Q][perf] n=$_statFrames '
                'avgFps=${_fpsAvg.toStringAsFixed(1)} '
                'conv=${(_sumConvMs / _statFrames).toStringAsFixed(1)}ms '
                'detect=${(_sumDetectMs / _statFrames).toStringAsFixed(1)}ms '
                'track=${(_sumTrackMs / _statFrames).toStringAsFixed(1)}ms | '
                'lbox=${s(0)} fill=${s(1)} send=${s(2)} run=${s(3)} '
                'recv=${s(4)} flat=${s(5)} post=${s(6)}',
          );
        }
      }

      if (mounted) {
        setState(() {
          _imageSize = Size(iw, ih);
          _detections = dets;
          _top = top;
          _fps = fpsNow;
        });
      }
    } catch (e) {
      // ignore: avoid_print
      print('[EYE-Q] frame error: $e');
    } finally {
      _busy = false;
    }
  }

  @override
  void dispose() {
    _camera?.stopImageStream();
    _camera?.dispose();
    _detector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_camera == null || !_camera!.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    // 상태바 높이: HUD가 시계/상태 아이콘과 겹치지 않도록 이만큼 아래로 내린다
    final double topInset = MediaQuery.of(context).padding.top;

    // HUD에 표시할 통계 문자열
    final String statText = _statFrames == 0
        ? 'warm-up $_frameCount/$_warmupFrames'
        : 'avg ${_fpsAvg.toStringAsFixed(1)} / '
        'min ${(_fpsMin.isInfinite ? 0 : _fpsMin).toStringAsFixed(1)} / '
        'max ${_fpsMax.toStringAsFixed(1)} (n=$_statFrames)';

    String stageText = '';
    String sub1 = '';
    String sub2 = '';
    if (_statFrames > 0) {
      final double c = _sumConvMs / _statFrames;
      final double d = _sumDetectMs / _statFrames;
      final double t = _sumTrackMs / _statFrames;
      stageText = 'ms  conv ${c.toStringAsFixed(0)} / '
          'detect ${d.toStringAsFixed(0)} / '
          'track ${t.toStringAsFixed(0)} '
          '(sum ${(c + d + t).toStringAsFixed(0)})';

      String f(int i) => (_sumSub[i] / _statFrames).toStringAsFixed(0);
      sub1 = 'detect: lbox ${f(0)} / fill ${f(1)} / send ${f(2)} / run ${f(3)}';
      sub2 = 'detect: recv ${f(4)} / flat ${f(5)} / post ${f(6)}';
    }

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_camera!),
          CustomPaint(
            painter: _HudPainter(
              detections: _detections,
              top: _top,
              imageSize: _imageSize,
              fps: _fps,
              topInset: topInset,
              statText: statText,
              stageText: stageText,
              subText1: sub1,
              subText2: sub2,
            ),
          ),
        ],
      ),
    );
  }
}

// ─── isolate로 넘길 순수 데이터만 담는 클래스 ───
// (CameraImage는 통째로 isolate에 못 넘기므로 바이트/치수만 추려서 담음)
class _YuvFrame {
  final Uint8List yBytes, uBytes, vBytes;
  final int width, height;
  final int yRowStride, uvRowStride, uvPixelStride;

  _YuvFrame({
    required this.yBytes,
    required this.uBytes,
    required this.vBytes,
    required this.width,
    required this.height,
    required this.yRowStride,
    required this.uvRowStride,
    required this.uvPixelStride,
  });
}

// ─── CameraImage(YUV420) → image.Image(RGB) ───
//   ⚠️ top-level 함수여야 compute()에서 별도 isolate로 실행 가능함
img.Image _convertYuv420(_YuvFrame f) {
  final image = img.Image(width: f.width, height: f.height);
  for (int y = 0; y < f.height; y++) {
    for (int x = 0; x < f.width; x++) {
      final int uvIndex =
          f.uvPixelStride * (x ~/ 2) + f.uvRowStride * (y ~/ 2);
      final int yIndex = y * f.yRowStride + x;
      final int yp = f.yBytes[yIndex];
      final int up = f.uBytes[uvIndex];
      final int vp = f.vBytes[uvIndex];
      final int r = (yp + 1.402 * (vp - 128)).round().clamp(0, 255);
      final int g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128))
          .round()
          .clamp(0, 255);
      final int b = (yp + 1.772 * (up - 128)).round().clamp(0, 255);
      image.setPixelRgb(x, y, r, g, b);
    }
  }
  return image;
}

// ─── 4-5: 박스 + HUD 그리기 ───
class _HudPainter extends CustomPainter {
  final List<Detection> detections;
  final Detection? top;
  final Size imageSize;
  final double fps;
  final double topInset; // 상태바 높이 (HUD 시작 위치)
  final String statText; // avg/min/max FPS 줄
  final String stageText; // conv/detect/track 줄
  final String subText1; // detect 내부 구간 1
  final String subText2; // detect 내부 구간 2

  _HudPainter({
    required this.detections,
    required this.top,
    required this.imageSize,
    required this.fps,
    required this.topInset,
    required this.statText,
    required this.stageText,
    required this.subText1,
    required this.subText2,
  });

  // HUD 영역 아래쪽 끝 (박스 라벨이 HUD와 겹치지 않게 하는 기준선)
  double get hudBottom => topInset + 148;

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize == Size.zero) return;
    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;

    // ── 박스 + 라벨 ──
    for (final d in detections) {
      final bool isTop = top != null && d.trackId == top!.trackId;
      final Color color =
      isTop ? const Color(0xFFFF0000) : const Color(0xFF969696);

      final rect = Rect.fromLTRB(
        d.box.left * scaleX,
        d.box.top * scaleY,
        d.box.right * scaleX,
        d.box.bottom * scaleY,
      );

      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = isTop ? 4 : 1;
      canvas.drawRect(rect, paint);

      final String label = isTop
          ? '!! THREAT: ${d.label}  '
          'TTC:${d.ttc.isInfinite ? "--" : d.ttc.toStringAsFixed(1)}s  '
          'Score:${d.score.toStringAsFixed(2)}'
          : '${kGradeName[d.grade]} ${d.label}';

      // 라벨 위치: 박스 위 → 헤더와 겹치면 박스 아래로 → 화면 밖이면 박스 안쪽
      double ly = rect.top - 16;
      if (ly < hudBottom) ly = rect.bottom + 2;
      if (ly + 16 > size.height) ly = rect.top + 2;
      _drawTextBg(canvas, label, Offset(rect.left, ly), color, 12);
    }

    // ── 상단 HUD (상태바 아래부터) ──
    final double t = topInset;
    _drawTextBg(canvas, 'FPS: ${fps.toStringAsFixed(0)}', Offset(8, t + 8),
        const Color(0xFFFFFFFF), 14);
    final String summary = top != null ? top!.summaryText : 'No threat';
    _drawTextBg(canvas, summary, Offset(8, t + 34),
        top != null ? const Color(0xFFFF0000) : const Color(0xFF00FF00), 14);

    // 누적 FPS 통계 (노란색)
    _drawTextBg(
        canvas, statText, Offset(8, t + 58), const Color(0xFFFFFF00), 12);
    // 구간별 소요시간 (하늘색)
    if (stageText.isNotEmpty) {
      _drawTextBg(
          canvas, stageText, Offset(8, t + 80), const Color(0xFF00FFFF), 12);
    }
    // detect 내부 구간 (주황색)
    if (subText1.isNotEmpty) {
      _drawTextBg(
          canvas, subText1, Offset(8, t + 102), const Color(0xFFFFB74D), 12);
    }
    if (subText2.isNotEmpty) {
      _drawTextBg(
          canvas, subText2, Offset(8, t + 124), const Color(0xFFFFB74D), 12);
    }
  }

  // 검은 배경 패치 위에 텍스트 (어떤 배경에서도 잘 보이게)
  void _drawTextBg(
      Canvas canvas, String text, Offset pos, Color color, double fontSize) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: color,
          fontSize: fontSize,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final double x = pos.dx < 0 ? 0.0 : pos.dx; // 화면 왼쪽으로 안 넘치게
    final bg = Paint()..color = const Color(0xCC000000);
    canvas.drawRect(
      Rect.fromLTWH(x, pos.dy, tp.width + 6, tp.height + 2),
      bg,
    );
    tp.paint(canvas, Offset(x + 3, pos.dy + 1));
  }

  @override
  bool shouldRepaint(covariant _HudPainter oldDelegate) => true;
}