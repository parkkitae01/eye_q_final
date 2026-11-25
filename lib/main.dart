// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : 메인 앱 (카메라 → YOLO → 트래커 → 소리·진동 경고 → 화면)
// ════════════════════════════════════════════════════════════════
// ⚠️ 수정 이력
//  1) YUV→RGB 변환을 compute()로 별도 isolate에서 실행 (UI 스레드 블로킹 제거)
//  2) FPS 누적 통계 + 워밍업 프레임 제외 + 구간별 소요시간(conv/detect/track)
//  3) detect 내부 구간(lbox/fill/send/run/recv/flat/post)까지 측정
//  4) 소리(TTS)+진동 경고 추가 (alert_manager.dart)
//  5) [2026-11-24] 접근성 안내: 시작/토글 음성
//  6) [2026-11-24] 발표용 미니멀 UI
//     - 평소: 깔끔한 화면(둥근 박스 + 하단 안내 카드 + 위험 시 테두리 강조)
//     - 오른쪽 위 버튼으로 디버깅 HUD(FPS·구간 시간 등) 전환 → 성능 측정은 그대로
// ════════════════════════════════════════════════════════════════

import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart'; // compute(), debugPrint
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import 'alert_manager.dart';
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
  final AlertManager _alert = AlertManager();

  List<Detection> _detections = [];
  Detection? _top;
  Size _imageSize = Size.zero; // 카메라 프레임 해상도 (좌표 변환용)
  double _fps = 0;
  bool _busy = false; // 추론 중이면 프레임 스킵 (밀림 방지)
  bool _debug = false; // [2026-11-24] 디버깅 HUD 표시 여부 (오른쪽 위 버튼)
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
    await _alert.init();
    _alert.announceStart(); // 시작 음성 안내
    _camera = CameraController(
      _cameras.first,
      ResolutionPreset.medium, // 화면 미리보기 화질 유지 (분석용 변환은 아래에서 절반 크기로)
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

      // ④ 소리·진동 경고 (now = 이 프레임이 도착한 시각 → 지연 측정 기준)
      _alert.update(top, iw, ih, now);

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
    _alert.dispose();
    super.dispose();
  }

  // 위험 단계별 강조 색
  Color _accentFor(AlertLevel level) {
    switch (level) {
      case AlertLevel.danger:
        return const Color(0xFFE5484D); // 빨강
      case AlertLevel.caution:
        return const Color(0xFFF0A92E); // 주황
      case AlertLevel.none:
        return const Color(0xFF32C759); // 초록(안전)
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_camera == null || !_camera!.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final double topInset = MediaQuery.of(context).padding.top;

    // 현재 최상위 위협의 경고 단계 (화면 색·아이콘용)
    final AlertLevel level =
    _top != null ? _alert.levelOf(_top!) : AlertLevel.none;
    final Color accent = _accentFor(level);
    final bool hasThreat = _top != null && level != AlertLevel.none;

    // 하단 안내 카드에 띄울 문구 (음성과 같은 내용)
    final String guide = _alert.lastText.isEmpty ? '대기 중' : _alert.lastText;

    return Scaffold(
      backgroundColor: Colors.black,
      body: GestureDetector(
        behavior: HitTestBehavior.opaque,
        // 화면 아무 곳이나 탭 → 경고 ON/OFF (음성 안내)
        onTap: () {
          setState(() => _alert.enabled = !_alert.enabled);
          _alert.announceToggle();
        },
        child: Stack(
          fit: StackFit.expand,
          children: [
            // 1) 카메라 미리보기
            CameraPreview(_camera!),

            // 2) 탐지 박스 (평소: 둥근 박스 / 디버그: 원본 라벨)
            CustomPaint(
              painter: _BoxPainter(
                detections: _detections,
                top: _top,
                imageSize: _imageSize,
                minimal: !_debug,
                topColor: accent,
              ),
            ),

            // 3) 위험 시 화면 테두리 강조
            if (hasThreat)
              IgnorePointer(
                child: Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: accent, width: 8),
                  ),
                ),
              ),

            // 4) 상단 바 (앱 이름 · 경고 on/off · 디버그 전환)
            Positioned(
              top: topInset + 8,
              left: 18,
              right: 12,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'EYE-Q',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1.5,
                    ),
                  ),
                  Row(
                    children: [
                      Icon(
                        _alert.enabled ? Icons.volume_up : Icons.volume_off,
                        color: _alert.enabled ? Colors.white : Colors.white54,
                        size: 24,
                      ),
                      IconButton(
                        onPressed: () => setState(() => _debug = !_debug),
                        icon: Icon(
                          Icons.tune,
                          color: _debug ? accent : Colors.white54,
                          size: 24,
                        ),
                        tooltip: '개발자 정보',
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // 5) 디버그 HUD (성능 측정용) — 버튼으로 켤 때만
            if (_debug)
              Positioned(
                top: topInset + 52,
                left: 12,
                child: _debugHud(),
              ),

            // 6) 하단 안내 카드 (음성과 같은 안내를 크게)
            Positioned(
              left: 20,
              right: 20,
              bottom: 40,
              child: Semantics(
                liveRegion: true,
                label: guide,
                child: Container(
                  padding:
                  const EdgeInsets.symmetric(horizontal: 24, vertical: 22),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.62),
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(
                      color: hasThreat ? accent : Colors.white24,
                      width: 2,
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        hasThreat
                            ? Icons.warning_amber_rounded
                            : Icons.check_circle_outline,
                        color: accent,
                        size: 40,
                      ),
                      const SizedBox(width: 18),
                      Expanded(
                        child: Text(
                          guide,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 26,
                            fontWeight: FontWeight.w700,
                            height: 1.25,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // 디버그용 성능 HUD (버튼으로 켤 때만 표시)
  Widget _debugHud() {
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

    final String alertText = _alert.enabled
        ? 'ALERT ON  last ${_alert.lastLatencyMs}ms  #${_alert.alertCount}'
        : 'ALERT OFF';

    Widget line(String s, Color c) => Text(
      s,
      style: TextStyle(color: c, fontSize: 12, fontWeight: FontWeight.bold),
    );

    return Container(
      padding: const EdgeInsets.all(8),
      color: const Color(0xCC000000),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          line('FPS: ${_fps.toStringAsFixed(0)}', Colors.white),
          line(_top != null ? _top!.summaryText : 'No threat',
              _top != null ? const Color(0xFFFF6B6B) : const Color(0xFF00FF88)),
          line(statText, const Color(0xFFFFEB3B)),
          if (stageText.isNotEmpty) line(stageText, const Color(0xFF4DD0E1)),
          if (sub1.isNotEmpty) line(sub1, const Color(0xFFFFB74D)),
          if (sub2.isNotEmpty) line(sub2, const Color(0xFFFFB74D)),
          line(alertText,
              _alert.enabled ? const Color(0xFF69F0AE) : Colors.white54),
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
  // 분석용이라 가로·세로를 절반으로 줄여서 변환 (2칸씩 건너뛰며 샘플링)
  // → 변환 비용이 약 1/4. 화면 미리보기(CameraPreview)는 원본 화질 그대로.
  final int w = f.width ~/ 2;
  final int h = f.height ~/ 2;
  final image = img.Image(width: w, height: h);
  for (int y = 0; y < h; y++) {
    final int yRow = (y * 2) * f.yRowStride;
    final int uvRow = f.uvRowStride * y; // (2y) ~/ 2 == y
    for (int x = 0; x < w; x++) {
      final int uvIndex = f.uvPixelStride * x + uvRow; // (2x) ~/ 2 == x
      final int yp = f.yBytes[yRow + x * 2];
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

// ─── 탐지 박스 그리기 ───
//   minimal=true  : 발표용 (둥근 박스만, 라벨은 하단 카드가 담당)
//   minimal=false : 디버그 (원본 사각 박스 + 영어 라벨/TTC/점수)
class _BoxPainter extends CustomPainter {
  final List<Detection> detections;
  final Detection? top;
  final Size imageSize;
  final bool minimal;
  final Color topColor;

  _BoxPainter({
    required this.detections,
    required this.top,
    required this.imageSize,
    required this.minimal,
    required this.topColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize == Size.zero) return;
    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;

    for (final d in detections) {
      final bool isTop = top != null && d.trackId == top!.trackId;
      final rect = Rect.fromLTRB(
        d.box.left * scaleX,
        d.box.top * scaleY,
        d.box.right * scaleX,
        d.box.bottom * scaleY,
      );

      if (minimal) {
        // 발표용: 둥근 박스. 최상위 위협만 강조, 나머지는 은은하게.
        final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(16));
        final paint = Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = isTop ? 5 : 2
          ..color = isTop ? topColor : Colors.white.withOpacity(0.35);
        canvas.drawRRect(rrect, paint);
      } else {
        // 디버그: 원본 사각 박스 + 라벨
        final Color color =
        isTop ? const Color(0xFFFF0000) : const Color(0xFF969696);
        final paint = Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = isTop ? 4 : 1
          ..color = color;
        canvas.drawRect(rect, paint);

        final String label = isTop
            ? '!! THREAT: ${d.label}  '
            'TTC:${d.ttc.isInfinite ? "--" : d.ttc.toStringAsFixed(1)}s  '
            'Score:${d.score.toStringAsFixed(2)}'
            : '${kGradeName[d.grade]} ${d.label}';
        final tp = TextPainter(
          text: TextSpan(
            text: label,
            style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.bold),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        double ly = rect.top - 16;
        if (ly < 0) ly = rect.bottom + 2;
        final double lx = rect.left < 0 ? 0 : rect.left;
        final bg = Paint()..color = const Color(0xCC000000);
        canvas.drawRect(
            Rect.fromLTWH(lx, ly, tp.width + 6, tp.height + 2), bg);
        tp.paint(canvas, Offset(lx + 3, ly + 1));
      }
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter oldDelegate) => true;
}