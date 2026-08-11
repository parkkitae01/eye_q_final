// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : 경량 객체 트래커 (블록 4-2)
// ════════════════════════════════════════════════════════════════
// Python model.track()이 주던 객체 ID를 IoU 매칭으로 직접 부여하고,
// 이전 프레임 높이와 비교해 TTC·위험도 점수까지 계산한다.
// 파일 위치: lib/object_tracker.dart
// ════════════════════════════════════════════════════════════════

import 'dart:math' as math;
import 'dart:ui';

import 'risk_engine.dart';

// ─────────────────────────────────────────────
// NMS 후 나온 "원시 탐지" — 다음 블록(4-3 ONNX 추론)이 만들어서 넘겨줌
//   아직 trackId·ttc·score는 없는 상태 (그건 트래커가 채움)
// ─────────────────────────────────────────────
class RawDetection {
  final Rect box;     // 픽셀 좌표 박스
  final int classId;  // COCO 클래스 번호
  final String label; // 클래스 이름 (예: 'person')

  const RawDetection({
    required this.box,
    required this.classId,
    required this.label,
  });
}

// ─────────────────────────────────────────────
// 내부 추적 상태 (한 물체의 직전 모습)
// ─────────────────────────────────────────────
class _Track {
  final int id;
  final int classId;
  Rect box;
  double height;
  int missed; // 연속으로 매칭 안 된 프레임 수

  _Track(this.id, this.classId, this.box, this.height, this.missed);
}

// ─────────────────────────────────────────────
// 객체 트래커
//   update()에 이번 프레임 탐지들을 넣으면,
//   ID·TTC·위험도 점수까지 채운 Detection 리스트를 돌려줌
// ─────────────────────────────────────────────
class ObjectTracker {
  final double iouThreshold; // 이 값 이상 겹쳐야 "같은 물체"로 인정
  final int maxMissed;       // 이 프레임 수 넘게 안 보이면 트랙 삭제

  int _nextId = 0;
  final List<_Track> _tracks = [];

  ObjectTracker({this.iouThreshold = 0.3, this.maxMissed = 5});

  List<Detection> update(
      List<RawDetection> raws,
      double dt,
      double frameWidth,
      ) {
    final results = <Detection>[];
    final newTracks = <_Track>[];
    final used = <int>{}; // 이미 매칭에 쓰인 기존 트랙 인덱스

    for (final raw in raws) {
      // 1) 같은 클래스 중 IoU가 가장 큰 기존 트랙 찾기
      double bestIou = iouThreshold;
      int bestIdx = -1;
      for (int i = 0; i < _tracks.length; i++) {
        if (used.contains(i)) continue;
        if (_tracks[i].classId != raw.classId) continue;
        final iou = _iou(_tracks[i].box, raw.box);
        if (iou >= bestIou) {
          bestIou = iou;
          bestIdx = i;
        }
      }

      final curHeight = raw.box.height;
      int trackId;
      double ttc;

      if (bestIdx >= 0) {
        // 2) 기존 물체와 매칭됨 → ID 승계 + TTC 계산
        final t = _tracks[bestIdx];
        used.add(bestIdx);
        trackId = t.id;
        ttc = estimateTtc(t.height, curHeight, dt);
        // 트랙 최신화
        t.box = raw.box;
        t.height = curHeight;
        t.missed = 0;
      } else {
        // 3) 처음 본 물체 → 새 ID 발급 (비교 대상 없으니 TTC 무한대)
        trackId = _nextId++;
        ttc = double.infinity;
        newTracks.add(_Track(trackId, raw.classId, raw.box, curHeight, 0));
      }

      // 4) 위험도 계산 (risk_engine.dart의 두뇌 사용)
      final grade = getRiskLevel(raw.classId);
      final dirW = directionWeight(raw.box.center.dx, frameWidth);
      final score = computeRiskScore(grade, ttc, dirW);

      results.add(Detection(
        box: raw.box,
        trackId: trackId,
        classId: raw.classId,
        label: raw.label,
        grade: grade,
        ttc: ttc,
        score: score,
      ));
    }

    // 5) 이번 프레임에 안 잡힌 기존 트랙 → missed 증가, 오래되면 삭제
    for (int i = 0; i < _tracks.length; i++) {
      if (!used.contains(i)) _tracks[i].missed++;
    }
    _tracks.removeWhere((t) => t.missed > maxMissed);
    _tracks.addAll(newTracks); // 새 물체는 다음 프레임부터 추적 대상

    return results;
  }

  // ─── IoU (두 박스가 얼마나 겹치는가, 0.0 ~ 1.0) ───
  double _iou(Rect a, Rect b) {
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

  // ─── 최우선 위험 1개 고르기 (Python의 top_threat) ───
  Detection? topThreat(List<Detection> dets) {
    Detection? top;
    for (final d in dets) {
      if (d.score <= 0) continue; // 위험 없음(TTC 무한대 등)은 제외
      if (top == null || d.score > top.score) top = d;
    }
    return top;
  }
}