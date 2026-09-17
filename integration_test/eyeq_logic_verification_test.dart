import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:image/image.dart' as img;

import 'package:eye_q_final/yolo_detector.dart';
import 'package:eye_q_final/object_tracker.dart';
import 'package:eye_q_final/risk_engine.dart';

Future<img.Image> _loadAsset(String path) async {
  final ByteData data = await rootBundle.load(path);
  final Uint8List bytes = data.buffer.asUint8List();
  var decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw Exception('이미지 디코딩 실패: $path');
  }
  decoded = img.bakeOrientation(decoded);
  return decoded;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // 영상별 프레임 개수 (zip 폴더 안 frame_NNN.jpg 개수 그대로)
  const Map<String, int> videoFrameCounts = {
    '천천히_걷기_1': 80,
    '천천히_걷기_1-2': 97,
    '보통_걷기_1': 69,
    '보통_걷기_1-2': 73,
    '조깅_1': 27,
    '조깅_1-2': 27,
  };
  const double frameInterval = 0.2; // 0.2초 간격 추출이므로

  group('EYE-Q 연속 프레임 검증', () {
    for (final entry in videoFrameCounts.entries) {
      final videoName = entry.key;
      final frameCount = entry.value;

      testWidgets('$videoName - 트래킹·TTC 검증', (tester) async {
        final detector = YoloDetector();
        await detector.init();
        expect(detector.isReady, true, reason: 'ONNX 모델 로드 실패');

        final tracker = ObjectTracker();
        int? mainId;
        int keptCount = 0;
        int totalWithDetection = 0;
        final List<double> ttcSeq = [];

        for (int i = 1; i <= frameCount; i++) {
          final fname = 'frame_${i.toString().padLeft(3, '0')}.jpg';
          final path = 'assets/test_images/$videoName/$fname';
          final frame = await _loadAsset(path);

          final raws = await detector.detect(frame);
          final t = (i - 1) * frameInterval;
          final dets = tracker.update(
            raws,
            t,
            frame.width.toDouble(),
            frame.height.toDouble(),
          );

          if (dets.isEmpty) continue;
          totalWithDetection++;

          dets.sort((a, b) => b.box.height.compareTo(a.box.height));
          final top = dets.first;
          mainId ??= top.trackId;

          if (dets.any((d) => d.trackId == mainId)) keptCount++;
          if (top.ttc.isFinite) ttcSeq.add(top.ttc);

          // ignore: avoid_print
          print('[$videoName] frame$i t=${t.toStringAsFixed(1)}s '
              'id=${top.trackId} ttc=${top.ttc.isFinite ? top.ttc.toStringAsFixed(2) : "--"} '
              'boxH=${top.box.height.toStringAsFixed(0)}');
        }

        final keepRate = totalWithDetection == 0 ? 0.0 : keptCount / totalWithDetection;
        // ignore: avoid_print
        print('📊 [$videoName] ID 유지율: ${(keepRate * 100).toStringAsFixed(1)}% ($keptCount/$totalWithDetection)');
        // ignore: avoid_print
        print('📊 [$videoName] TTC 추이: $ttcSeq');

        expect(totalWithDetection > 0, true, reason: '$videoName: 탐지가 한 번도 안 됨');
        expect(keepRate >= 0.8, true,
            reason: '$videoName: ID 유지율 ${(keepRate*100).toStringAsFixed(0)}% (목표 80% 미달)');

        await detector.dispose();
      });
    }
  });
}