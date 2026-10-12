// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : YOLOv8n ONNX 탐지기 (블록 4-3)  — v4 속도 개선판
// ════════════════════════════════════════════════════════════════
// 파일 위치: lib/yolo_detector.dart
//
// v5 변경점
//   - 입력 크기 640 → 320 (모델도 imgsz=320으로 다시 export 해야 함)
//   - 카메라 프레임 회전 보정: detect(img, rotation: sensorOrientation)
//     (카메라 원본은 가로로 누운 상태라 모델이 옆으로 누운 세상을 보고 있었음)
//   - XNNPACK은 kUseXnnpack 플래그로 끄고 켤 수 있게 함 (기본 꺼짐)
// v4 변경점 (측정 결과 기반: detect 553ms 중 전처리 176 + 출력 변환 203ms가 낭비)
//   1) 전처리: image 패키지의 copyResize/fill/compositeImage/getPixel 대신
//      원본 바이트에서 바로 640x640 텐서로 샘플링 (결과는 기존과 동일: nearest + 회색 114 패딩)
//   2) 후처리: 70만 개짜리 중첩 리스트를 flatten 하지 않고 행 단위로 바로 읽음
//   3) ONNX 세션: XNNPACK + 4스레드 시도 (실패하면 기본 설정으로 자동 대체)
//   4) 프레임마다 찍던 print 제거
// DetectTimings 필드 이름은 그대로라서 main.dart는 수정할 필요 없음
//   lbox = 바이트·좌표표 준비, fill = 텐서 채우기, flat = 출력 행 추출
// ════════════════════════════════════════════════════════════════

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'object_tracker.dart'; // RawDetection

// 모델 입력 크기 — ONNX export 시 imgsz와 같아야 함 (이번엔 320으로 export)
const int kInputSize = 320;
const bool kUseXnnpack = false; // true면 XNNPACK + 4스레드 시도
const double kConfThreshold = 0.4; // 이 점수 미만 후보는 버림
const double kNmsIou = 0.45; // 이 이상 겹치면 중복으로 제거

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

// detect() 내부 구간별 소요 시간(ms) — main.dart HUD가 읽어감
class DetectTimings {
  double letterbox = 0;
  double tensorFill = 0;
  double send = 0;
  double run = 0;
  double receive = 0;
  double flatten = 0;
  double post = 0;
}

class YoloDetector {
  OnnxRuntime? _ort;
  dynamic _session; // OrtSession (타입 추론 회피용 dynamic)
  String _inputName = '';
  String _outputName = '';
  bool _ready = false;

  final DetectTimings timings = DetectTimings();

  bool get isReady => _ready;

  // ─── 모델 로드 ───
  Future<void> init() async {
    _ort = OnnxRuntime();
    const modelAsset = 'assets/models/yolov8n.onnx';
    if (kUseXnnpack) {
      try {
        _session = await _ort!.createSessionFromAsset(
          modelAsset,
          options: OrtSessionOptions(
            providers: [OrtProvider.XNNPACK, OrtProvider.CPU],
            intraOpNumThreads: 4,
          ),
        );
        // ignore: avoid_print
        print('[YOLO] session: XNNPACK + 4 threads');
      } catch (e) {
        // ignore: avoid_print
        print('[YOLO] XNNPACK 세션 실패 → 기본 설정으로 재시도: $e');
        _session = await _ort!.createSessionFromAsset(modelAsset);
      }
    } else {
      _session = await _ort!.createSessionFromAsset(modelAsset);
      // ignore: avoid_print
      print('[YOLO] session: 기본 설정');
    }
    _inputName = _session.inputNames.first as String;
    _outputName = _session.outputNames.first as String;
    _ready = true;
    // ignore: avoid_print
    print('[YOLO] inputs=${_session.inputNames} outputs=${_session.outputNames}');
  }

  // ─── 중첩 리스트를 1차원 숫자 리스트로 펴기 (예상 밖 구조일 때만 쓰는 대비용) ───
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

  // ─── 추론: 이미지 한 장 → 탐지 목록 ───
  // rotation: 카메라 sensorOrientation (0/90/180/270). 이미지를 시계방향으로 그만큼 돌려서 세움.
  // 반환되는 박스 좌표는 '세워진 이미지' 기준 (90/270이면 가로·세로가 서로 바뀐 크기).
  Future<List<RawDetection>> detect(img.Image src, {int rotation = 0}) async {
    if (!_ready) return [];

    final sw = Stopwatch()..start();
    double mark() {
      final ms = sw.elapsedMicroseconds / 1000.0;
      sw.reset();
      return ms;
    }

    // 1) letterbox 계산 (비율 유지 + 회색 여백) — '세워진' 이미지 크기 기준
    final int sw0 = src.width;
    final int sh0 = src.height;
    final int rot = ((rotation % 360) + 360) % 360;
    final bool swap = rot == 90 || rot == 270;
    final int uw = swap ? sh0 : sw0; // 세워진 이미지 너비
    final int uh = swap ? sw0 : sh0; // 세워진 이미지 높이
    final double scale = math.min(kInputSize / uw, kInputSize / uh);
    final int newW = (uw * scale).round();
    final int newH = (uh * scale).round();
    final int padX = ((kInputSize - newW) / 2).round();
    final int padY = ((kInputSize - newH) / 2).round();

    // 원본 픽셀 바이트 (RGB 3채널 uint8이면 길이 = w*h*3)
    final Uint8List bytes = src.getBytes();
    final int stride = bytes.length ~/ (sw0 * sh0);
    final bool fast = stride == 3 || stride == 4;

    // 세워진 좌표(ux,uy) → 원본 픽셀 번호(p = sy*sw0 + sx)
    //   회전해도 p = axP[x] + byP[y] 형태로 나뉘어서 안쪽 루프가 가벼움
    final axP = Int32List(newW);
    for (int x = 0; x < newW; x++) {
      final int ux = math.min(uw - 1, (x * uw / newW).floor());
      switch (rot) {
        case 90: // sx = uy, sy = sh0-1-ux
          axP[x] = (sh0 - 1 - ux) * sw0;
          break;
        case 180: // sx = sw0-1-ux
          axP[x] = sw0 - 1 - ux;
          break;
        case 270: // sy = ux
          axP[x] = ux * sw0;
          break;
        default: // 0: sx = ux
          axP[x] = ux;
      }
    }
    final byP = Int32List(newH);
    for (int y = 0; y < newH; y++) {
      final int uy = math.min(uh - 1, (y * uh / newH).floor());
      switch (rot) {
        case 90: // sx = uy
          byP[y] = uy;
          break;
        case 180: // sy = sh0-1-uy
          byP[y] = (sh0 - 1 - uy) * sw0;
          break;
        case 270: // sx = sw0-1-uy
          byP[y] = sw0 - 1 - uy;
          break;
        default: // 0: sy = uy
          byP[y] = uy * sw0;
      }
    }
    timings.letterbox = mark();

    // 2) [1,3,S,S] float32, RGB, 0~1 (CHW). 여백은 회색 114/255로 채움
    final int area = kInputSize * kInputSize;
    final input = Float32List(3 * area);
    input.fillRange(0, 3 * area, 114 / 255.0);
    const double inv = 1 / 255.0;

    for (int y = 0; y < newH; y++) {
      final int outRow = (y + padY) * kInputSize + padX;
      final int by = byP[y];
      if (fast) {
        for (int x = 0; x < newW; x++) {
          final int si = (axP[x] + by) * stride;
          final int o = outRow + x;
          input[o] = bytes[si] * inv; // R
          input[area + o] = bytes[si + 1] * inv; // G
          input[2 * area + o] = bytes[si + 2] * inv; // B
        }
      } else {
        // 예상 밖 이미지 형식일 때의 느린 대비 경로
        for (int x = 0; x < newW; x++) {
          final int pIdx = axP[x] + by;
          final px = src.getPixel(pIdx % sw0, pIdx ~/ sw0);
          final int o = outRow + x;
          input[o] = px.r / 255.0;
          input[area + o] = px.g / 255.0;
          input[2 * area + o] = px.b / 255.0;
        }
      }
    }
    timings.tensorFill = mark();

    // 3) 추론
    final inputOrt =
    await OrtValue.fromList(input, [1, 3, kInputSize, kInputSize]);
    timings.send = mark();

    final outputs = await _session.run({_inputName: inputOrt});
    timings.run = mark();

    final rawOutput = await outputs[_outputName]!.asList();
    timings.receive = mark();

    // 4) 출력 [1, 84, N]을 84개의 행으로 나눔 (N은 보통 8400)
    //    행 0~3 = cx,cy,w,h  /  행 4~83 = 클래스 점수 80개
    const int numClasses = 80;
    const int numRows = 4 + numClasses;
    List<List<dynamic>> rows;
    final dynamic o = rawOutput;
    if (o is List &&
        o.length == 1 &&
        o[0] is List &&
        (o[0] as List).length == numRows &&
        (o[0] as List)[0] is List) {
      final List<dynamic> r0 = o[0] as List;
      rows = List<List<dynamic>>.generate(
          numRows, (i) => r0[i] as List<dynamic>);
    } else {
      final flat = _flattenToNums(o);
      final int n = flat.length ~/ numRows;
      rows = List<List<dynamic>>.generate(
          numRows, (c) => flat.sublist(c * n, (c + 1) * n));
    }
    final int numBoxes = rows[0].length;
    timings.flatten = mark();

    // 5) 후처리: 박스마다 최고 클래스 점수 찾기 (행 단위로 훑어서 접근이 빠름)
    final best = Float32List(numBoxes);
    final bestCls = Int16List(numBoxes)..fillRange(0, numBoxes, -1);
    for (int c = 0; c < numClasses; c++) {
      final List<dynamic> row = rows[4 + c];
      for (int a = 0; a < numBoxes; a++) {
        final double s = (row[a] as num).toDouble();
        if (s > best[a]) {
          best[a] = s;
          bestCls[a] = c;
        }
      }
    }

    final cands = <_Cand>[];
    final List<dynamic> rCx = rows[0];
    final List<dynamic> rCy = rows[1];
    final List<dynamic> rW = rows[2];
    final List<dynamic> rH = rows[3];
    for (int a = 0; a < numBoxes; a++) {
      if (best[a] < kConfThreshold || bestCls[a] < 0) continue;

      // 박스 (cx,cy,w,h, 640=letterbox 기준) → 원본 좌표 (letterbox 역변환)
      final double cx = (rCx[a] as num).toDouble();
      final double cy = (rCy[a] as num).toDouble();
      final double w = (rW[a] as num).toDouble();
      final double h = (rH[a] as num).toDouble();

      final double x1 = (cx - w / 2 - padX) / scale;
      final double y1 = (cy - h / 2 - padY) / scale;
      final double x2 = (cx + w / 2 - padX) / scale;
      final double y2 = (cy + h / 2 - padY) / scale;

      cands.add(_Cand(Rect.fromLTRB(x1, y1, x2, y2), bestCls[a], best[a]));
    }

    // 6) NMS (클래스별 중복 박스 제거)
    final kept = _nms(cands, kNmsIou);

    // 7) 메모리 정리
    inputOrt.dispose();
    for (final out in outputs.values) {
      out.dispose();
    }
    timings.post = mark();

    // 8) RawDetection으로 변환해서 반환
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