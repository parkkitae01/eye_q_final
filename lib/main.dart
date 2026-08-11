// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : 메인 앱 (블록 4-4 카메라 연결 + 4-5 화면 그리기)
// ════════════════════════════════════════════════════════════════
// 카메라 스트림 → YOLO 추론 → 트래커(ID·TTC) → CustomPaint HUD
// 기존 lib/main.dart 내용을 전부 지우고 이걸로 교체하세요.
// ════════════════════════════════════════════════════════════════

import 'package:camera/camera.dart';
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
      final rgb = _yuv420ToImage(image);
      final raws = await _detector.detect(rgb);
      final dets = _tracker.update(raws, dt, rgb.width.toDouble());
      final top = _tracker.topThreat(dets);
      if (mounted) {
        setState(() {
          _imageSize = Size(rgb.width.toDouble(), rgb.height.toDouble());
          _detections = dets;
          _top = top;
          _fps = dt > 0 ? 1 / dt : 0;
        });
      }
    } catch (e) {
      // ignore: avoid_print
      print('[EYE-Q] frame error: $e');
    } finally {
      _busy = false;
    }
  }

  // ─── CameraImage(YUV420) → image.Image(RGB) ───
  //   Android 카메라 기본 포맷. 폰에서 가장 디버깅이 필요한 부분.
  img.Image _yuv420ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    final yPlane = cameraImage.planes[0];
    final uPlane = cameraImage.planes[1];
    final vPlane = cameraImage.planes[2];
    final int uvRowStride = uPlane.bytesPerRow;
    final int uvPixelStride = uPlane.bytesPerPixel ?? 1;

    final image = img.Image(width: width, height: height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex =
            uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
        final int yIndex = y * yPlane.bytesPerRow + x;
        final int yp = yPlane.bytes[yIndex];
        final int up = uPlane.bytes[uvIndex];
        final int vp = vPlane.bytes[uvIndex];
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
            ),
          ),
        ],
      ),
    );
  }
}

// ─── 4-5: 박스 + HUD 그리기 (Python 시각화 그대로 이식) ───
class _HudPainter extends CustomPainter {
  final List<Detection> detections;
  final Detection? top;
  final Size imageSize;
  final double fps;

  _HudPainter({
    required this.detections,
    required this.top,
    required this.imageSize,
    required this.fps,
  });

  static const double hudBottom = 65; // 상단 헤더 높이 (라벨 겹침 방지)

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

    // ── 상단 HUD ──
    _drawTextBg(canvas, 'FPS: ${fps.toStringAsFixed(0)}',
        const Offset(8, 8), const Color(0xFFFFFFFF), 14);
    final String summary = top != null ? top!.summaryText : 'No threat';
    _drawTextBg(canvas, summary, const Offset(8, 34),
        top != null ? const Color(0xFFFF0000) : const Color(0xFF00FF00), 14);
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