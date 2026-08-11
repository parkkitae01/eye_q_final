// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : YOLOv8n ONNX 탐지기 (블록 4-3)
// ════════════════════════════════════════════════════════════════
// 모델 로드 + 카메라 이미지 전처리 + 추론 + 후처리(NMS)
//   입력 : image 패키지의 Image 한 장
//   출력 : List<RawDetection>  (4-2 트래커가 받아서 ID·TTC를 채움)
// 파일 위치: lib/yolo_detector.dart
//
// ⚠️ 2026-07-29 수정: letterbox(비율 유지) 리사이즈 적용
//    기존엔 원본 비율 무시하고 640x640으로 강제로 눌러서 리사이즈했었는데,
//    이러면 세로로 긴 사진이 찌그러져서 모델이 클래스를 헷갈리는 문제가 있었음
//    (예: 멀리서 찍은 트럭이 보트로 오인식). 비율 유지 + 회색 여백 패딩으로 해결.
// ════════════════════════════════════════════════════════════════

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'object_tracker.dart'; // RawDetection

// 모델 입력 크기 — ONNX export 시 imgsz와 같아야 함 (ultralytics 기본 640)
const int kInputSize = 640;
const double kConfThreshold = 0.4;  // 이 점수 미만 후보는 버림
const double kNmsIou = 0.45;        // 이 이상 겹치면 중복으로 제거

// COCO 80 클래스 이름 (label 표시용)
const List<String> kCocoLabels = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush',
];

// NMS 내부용 후보
class _Cand {
  final Rect box;
  final int classId;
  final double score;
  _Cand(this.box, this.classId, this.score);
}

// letterbox 리사이즈 결과(스케일·패딩 정보까지 같이 보관)
class _LetterboxResult {
  final img.Image image;
  final double scale;   // 원본 → 리사이즈된 크기로의 배율
  final double padX;    // 좌우 패딩(픽셀, 640 기준)
  final double padY;    // 상하 패딩(픽셀, 640 기준)
  _LetterboxResult(this.image, this.scale, this.padX, this.padY);
}

class YoloDetector {
  double _debugMaxScore = 0; // ← 검증용 디버그 변수
  OnnxRuntime? _ort;
  dynamic _session; // OrtSession (타입 추론 회피용 dynamic)
  String _inputName = '';
  String _outputName = '';
  bool _ready = false;

  bool get isReady => _ready;

  // ─── 모델 로드 ───
  Future<void> init() async {
    _ort = OnnxRuntime();
    _session = await _ort!.createSessionFromAsset('assets/models/yolov8n.onnx');
    _inputName = _session.inputNames.first as String;
    _outputName = _session.outputNames.first as String;
    _ready = true;
    // 디버깅: 실제 모델 입출력 이름을 콘솔에 찍어 우리 가정과 대조
    // ignore: avoid_print
    print('[YOLO] inputs=${_session.inputNames} outputs=${_session.outputNames}');
  }

  // ─── 중첩 리스트([[[...]]] 등)를 1차원 숫자 리스트로 평평하게 펴기 ───
  //   flutter_onnxruntime의 asList()가 모델 output shape([1,84,8400]) 그대로
  //   중첩 리스트 구조로 돌려주기 때문에 필요함 (클래스 안에 있어야 detect()에서 호출 가능)
  List<num> _flattenToNums(dynamic nested) {
    final result = <num>[];
    void rec(dynamic e) {
      if (e is List) {
        for (final item in e) {
          rec(item);
        }
      } else if (e is num) {
        result.add(e);
      }
    }
    rec(nested);
    return result;
  }

  // ─── letterbox 리사이즈: 원본 비율을 유지한 채 640x640 안에 맞추고
  //     남는 여백은 회색(114,114,114)으로 채운다 (YOLO 계열 표준 전처리) ───
  _LetterboxResult _letterbox(img.Image src, int targetSize) {
    final double scale = math.min(
      targetSize / src.width,
      targetSize / src.height,
    );
    final int newW = (src.width * scale).round();
    final int newH = (src.height * scale).round();

    final resized = img.copyResize(src, width: newW, height: newH);

    // 640x640 회색 캔버스 생성
    final canvas = img.Image(width: targetSize, height: targetSize);
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));

    final int padX = ((targetSize - newW) / 2).round();
    final int padY = ((targetSize - newH) / 2).round();

    img.compositeImage(canvas, resized, dstX: padX, dstY: padY);

    return _LetterboxResult(canvas, scale, padX.toDouble(), padY.toDouble());
  }

  // ─── 추론: 이미지 한 장 → 탐지 목록 ───
  Future<List<RawDetection>> detect(img.Image src) async {
    if (!_ready) return [];

    // 1) letterbox 리사이즈 (비율 유지 + 회색 여백)
    final lb = _letterbox(src, kInputSize);
    final resized = lb.image;

    // 2) [1,3,640,640] float32, RGB, 0~1 정규화 (CHW: R 전체 → G 전체 → B 전체)
    final input = Float32List(3 * kInputSize * kInputSize);
    final area = kInputSize * kInputSize;
    int p = 0;
    for (int y = 0; y < kInputSize; y++) {
      for (int x = 0; x < kInputSize; x++) {
        final px = resized.getPixel(x, y);
        input[p] = px.r / 255.0;            // R
        input[area + p] = px.g / 255.0;     // G
        input[2 * area + p] = px.b / 255.0; // B
        p++;
      }
    }

    // 3) 추론
    final inputOrt = await OrtValue.fromList(input, [1, 3, kInputSize, kInputSize]);
    final outputs = await _session.run({_inputName: inputOrt});
    final rawOutput = await outputs[_outputName]!.asList();
    final flat = _flattenToNums(rawOutput);

    // ignore: avoid_print
    print('[디버그] 출력 데이터 총 개수: ${flat.length} (84로 나누면: ${flat.length / 84})');

    // 4) 후처리: YOLOv8 출력 [1, 84, 8400] (84 = 4박스 + 80클래스)
    //    flat 인덱싱: 값[채널 * numBoxes + 박스인덱스]
    const int numClasses = 80;
    final int numBoxes = flat.length ~/ (4 + numClasses); // 보통 8400
    final cands = <_Cand>[];

    for (int a = 0; a < numBoxes; a++) {
      // 최대 클래스 점수 찾기
      double best = 0;
      int bestCls = -1;
      for (int c = 0; c < numClasses; c++) {
        final s = flat[(4 + c) * numBoxes + a].toDouble();
        if (s > best) {
          best = s;
          bestCls = c;
        }
      }
      if (best > _debugMaxScore) _debugMaxScore = best; // ← 검증용
      if (best < kConfThreshold || bestCls < 0) continue;

      // 박스 (cx,cy,w,h, 640=letterbox 기준) → 원본 좌표 (x1,y1,x2,y2)
      //   letterbox 역변환: (640좌표 - 패딩) / scale
      final cx = flat[a].toDouble();
      final cy = flat[numBoxes + a].toDouble();
      final w = flat[2 * numBoxes + a].toDouble();
      final h = flat[3 * numBoxes + a].toDouble();

      final double x1 = (cx - w / 2 - lb.padX) / lb.scale;
      final double y1 = (cy - h / 2 - lb.padY) / lb.scale;
      final double x2 = (cx + w / 2 - lb.padX) / lb.scale;
      final double y2 = (cy + h / 2 - lb.padY) / lb.scale;

      cands.add(_Cand(
        Rect.fromLTRB(x1, y1, x2, y2),
        bestCls,
        best,
      ));
    }

    // ignore: avoid_print
    print('[디버그] 이 프레임 최고 신뢰도 점수: $_debugMaxScore (임계값: $kConfThreshold)'); // ← 검증용

    // 5) NMS (클래스별 중복 박스 제거)
    final kept = _nms(cands, kNmsIou);

    // 6) 메모리 정리
    inputOrt.dispose();
    for (final o in outputs.values) {
      o.dispose();
    }

    // 7) RawDetection으로 변환해서 반환
    return kept
        .map((c) => RawDetection(
      box: c.box,
      classId: c.classId,
      label: c.classId < kCocoLabels.length
          ? kCocoLabels[c.classId]
          : 'obj',
    ))
        .toList();
  }

  // ─── NMS (Non-Maximum Suppression) ───
  List<_Cand> _nms(List<_Cand> cands, double iouTh) {
    cands.sort((a, b) => b.score.compareTo(a.score)); // 점수 높은 순
    final removed = List<bool>.filled(cands.length, false);
    final kept = <_Cand>[];
    for (int i = 0; i < cands.length; i++) {
      if (removed[i]) continue;
      kept.add(cands[i]);
      for (int j = i + 1; j < cands.length; j++) {
        if (removed[j]) continue;
        if (cands[i].classId == cands[j].classId &&
            _boxIou(cands[i].box, cands[j].box) > iouTh) {
          removed[j] = true;
        }
      }
    }
    return kept;
  }

  double _boxIou(Rect a, Rect b) {
    final x1 = math.max(a.left, b.left);
    final y1 = math.max(a.top, b.top);
    final x2 = math.min(a.right, b.right);
    final y2 = math.min(a.bottom, b.bottom);
    final w = math.max(0.0, x2 - x1);
    final h = math.max(0.0, y2 - y1);
    final inter = w * h;
    final union = a.width * a.height + b.width * b.height - inter;
    return union <= 0 ? 0.0 : inter / union;
  }

  // ─── 정리 ───
  Future<void> dispose() async {
    if (_session != null) {
      await _session.close();
      _session = null;
    }
    _ready = false;
  }
}
