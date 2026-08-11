// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : 로직 검증 통합 테스트
// ════════════════════════════════════════════════════════════════
// 목적: 카메라/실기기 없이도 YOLO 탐지 → 객체 트래킹 → TTC/위험도
//       계산까지 전체 파이프라인이 올바르게 동작하는지 검증
//
// 실행 위치: <프로젝트 루트>/integration_test/eyeq_logic_verification_test.dart
// 실행 명령: flutter test integration_test/eyeq_logic_verification_test.dart -d <에뮬레이터ID>
//
// 사전 준비물 (아래 "설치 가이드" 참고):
//   1) integration_test 패키지 추가
//   2) 테스트용 이미지 2장을 assets/test_images/ 에 추가
//      - frame1.jpg : 물체(예: 사람)가 멀리 있는 사진
//      - frame2.jpg : 같은 물체가 더 가까이(=박스가 더 크게) 찍힌 사진
//   3) pubspec.yaml assets에 test_images 폴더 등록
// ════════════════════════════════════════════════════════════════

import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:image/image.dart' as img;

// ⚠️ 아래 3줄의 패키지명(eyeq_final)은 실제 pubspec.yaml의 name: 값으로 바꿔주세요.
import 'package:eye_q_final/yolo_detector.dart';
import 'package:eye_q_final/object_tracker.dart';
import 'package:eye_q_final/risk_engine.dart';
Future<img.Image> _loadAsset(String path) async {
  final ByteData data = await rootBundle.load(path);
  final Uint8List bytes = data.buffer.asUint8List();
  var decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw Exception('이미지 디코딩 실패: $path (파일 손상 또는 지원 안 되는 형식)');
  }
  // 핸드폰 사진은 EXIF에 회전 정보만 따로 붙는 경우가 많아,
  // 실제 픽셀을 보이는 방향대로 회전시켜준다. (안 하면 YOLO가 옆으로 누운 사진으로 인식)
  decoded = img.bakeOrientation(decoded);
  // ignore: avoid_print
  print('[디버그] $path 보정 후 크기: ${decoded.width}x${decoded.height}');
  return decoded;
}

void _printDetection(String tag, Detection d) {
  // ignore: avoid_print
  print(
    '[$tag] id=${d.trackId} label=${d.label} grade=${kGradeName[d.grade]} '
    'ttc=${d.ttc.isInfinite ? "--" : d.ttc.toStringAsFixed(2)}s '
    'score=${d.score.toStringAsFixed(3)} '
    'box=(${d.box.left.toStringAsFixed(0)},${d.box.top.toStringAsFixed(0)},'
    '${d.box.right.toStringAsFixed(0)},${d.box.bottom.toStringAsFixed(0)})',
  );
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('EYE-Q 로직 검증', () {
    testWidgets('YOLO 탐지 → 트래커 → TTC 파이프라인 검증', (tester) async {
      // ── 1) YOLO 모델 로드 ──
      final detector = YoloDetector();
      await detector.init();
      expect(detector.isReady, true, reason: 'ONNX 모델 로드 실패');
      // ignore: avoid_print
      print('✅ 1) YOLOv8n 모델 로드 성공');

      // ── 2) 테스트 이미지 2장 로드 ──
      final frame1 = await _loadAsset('assets/test_images/frame1.jpg');
      final frame2 = await _loadAsset('assets/test_images/frame2.jpg');
      // ignore: avoid_print
      print('✅ 2) 테스트 이미지 로드 완료 '
          '(frame1: ${frame1.width}x${frame1.height}, '
          'frame2: ${frame2.width}x${frame2.height})');

      // ── 3) 프레임별 YOLO 추론 ──
      final raws1 = await detector.detect(frame1);
      final raws2 = await detector.detect(frame2);
      // ignore: avoid_print
      print('✅ 3) 탐지 결과: frame1=${raws1.length}개, frame2=${raws2.length}개');
      expect(raws1.isNotEmpty, true, reason: '첫 프레임에서 아무것도 탐지되지 않음 → 테스트 이미지를 확인하세요');
      expect(raws2.isNotEmpty, true, reason: '두번째 프레임에서 아무것도 탐지되지 않음 → 테스트 이미지를 확인하세요');

      for (final r in raws1) {
        // ignore: avoid_print
        print('  frame1 raw: ${r.label}(class ${r.classId}) box=${r.box}');
      }
      for (final r in raws2) {
        // ignore: avoid_print
        print('  frame2 raw: ${r.label}(class ${r.classId}) box=${r.box}');
      }

      // ── 4) 트래커에 순서대로 투입 (dt=0.5초 가정, 실제 두 사진 촬영 간격에 맞게 조정) ──
      const double dt = 0.5;
      final tracker = ObjectTracker();

      final dets1 = tracker.update(raws1, 0.0, frame1.width.toDouble());
      // ignore: avoid_print
      print('--- Frame 1 결과 ---');
      for (final d in dets1) {
        _printDetection('F1', d);
      }

      final dets2 = tracker.update(raws2, dt, frame2.width.toDouble());
      // ignore: avoid_print
      print('--- Frame 2 결과 (TTC 계산 대상) ---');
      for (final d in dets2) {
        _printDetection('F2', d);
      }

      // ── 5) 검증 포인트 ──
      // 5-1) 같은 물체는 frame1과 frame2에서 동일 trackId를 유지해야 함
      final ids1 = dets1.map((d) => d.trackId).toSet();
      final ids2 = dets2.map((d) => d.trackId).toSet();
      final matchedIds = ids1.intersection(ids2);
      // ignore: avoid_print
      print('🔗 두 프레임에서 매칭 유지된 trackId: $matchedIds');
      expect(matchedIds.isNotEmpty, true,
          reason: 'IoU 매칭이 안 됨 → iouThreshold(0.3)나 테스트 이미지의 프레이밍을 확인하세요');

      // 5-2) 물체가 가까워졌다면(=박스가 커졌다면) TTC가 유한값(양수)이어야 함
      for (final id in matchedIds) {
        final before = dets1.firstWhere((d) => d.trackId == id);
        final after = dets2.firstWhere((d) => d.trackId == id);
        final grew = after.box.height > before.box.height;
        // ignore: avoid_print
        print('  id=$id  height ${before.box.height.toStringAsFixed(1)} → '
            '${after.box.height.toStringAsFixed(1)} '
            '(${grew ? "커짐→TTC 유한값 기대" : "안커짐→TTC 무한대 기대"}) '
            '실제 ttc=${after.ttc}');
        if (grew) {
          expect(after.ttc.isFinite, true,
              reason: 'id=$id 박스가 커졌는데 TTC가 무한대로 나옴 → estimateTtc 로직 확인 필요');
          expect(after.ttc > 0, true);
        } else {
          expect(after.ttc.isInfinite, true,
              reason: 'id=$id 박스가 안 커졌는데 TTC가 유한값으로 나옴 → estimateTtc 로직 확인 필요');
        }
      }

      // 5-3) 최우선 위험 객체 산출 확인
      final top = tracker.topThreat(dets2);
      // ignore: avoid_print
      print(top != null
          ? '🚨 최우선 위험: ${top.label} (score=${top.score.toStringAsFixed(3)})'
          : 'ℹ️ 최우선 위험 없음 (모든 객체 score<=0)');

      await detector.dispose();
      // ignore: avoid_print
      print('✅ 검증 완료');
    });
  });
}
