// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : 경량 객체 트래커 (블록 4-2)
// ════════════════════════════════════════════════════════════════
// Python model.track()이 주던 객체 ID를 IoU 매칭으로 직접 부여하고,
// 최근 프레임 높이 기록(추세선)으로 TTC·위험도 점수까지 계산한다.
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
// 내부 추적 상태 (한 물체의 최근 모습)
// ─────────────────────────────────────────────
class _Track {
  final int id;
  final int classId;
  Rect box;
  final List<(double, double)> heightHistory; // (누적시간, 박스높이) 최근 5개
  int missed; // 연속으로 매칭 안 된 프레임 수

  _Track(this.id, this.classId, this.box, double height, double t, this.missed)
      : heightHistory = [(t, height)];

  void addHeight(double t, double height) {
    heightHistory.add((t, height));
    if (heightHistory.length > 5) heightHistory.removeAt(0);
  }
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

  // ⚠️ t = 추적 시작 시점부터 누적된 시간(초). 프레임 간 간격(delta)이 아님.
  // ⚠️ frameHeight 추가: risk_engine.dart의 근접 안전장치(resolveRisk)가
  //    "박스 높이가 화면의 80% 이상인지"를 판단하려면 화면 세로 길이가 필요함.
  //    → 호출부(main.dart 등)에서 반드시 인자를 하나 더 넘겨줘야 함.
  List<Detection> update(
      List<RawDetection> raws,
      double t,
      double frameWidth,
      double frameHeight,
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
      List<(double, double)> heightHistoryForRisk;

      if (bestIdx >= 0) {
        // 2) 기존 물체와 매칭됨 → ID 승계 + 기록 추가
        final tr = _tracks[bestIdx];
        used.add(bestIdx);
        trackId = tr.id;
        tr.addHeight(t, curHeight);
        heightHistoryForRisk = tr.heightHistory;
        tr.box = raw.box;
        tr.missed = 0;
      } else {
        // 3) 처음 본 물체 → 새 ID 발급 (기록 1개뿐이니 TTC는 resolveRisk 내부에서 무한대 처리)
        trackId = _nextId++;
        final newTrack = _Track(trackId, raw.classId, raw.box, curHeight, t, 0);
        newTracks.add(newTrack);
        heightHistoryForRisk = newTrack.heightHistory;
      }

      // 4) 위험도 계산 (risk_engine.dart의 두뇌 사용)
      //    - 기존: getRiskLevel() + estimateTtc() + computeRiskScore()를 각각 호출
      //    - 변경: resolveRisk() 하나로 통합, 근접(박스 80% 이상) 시 강제 CRITICAL 처리 포함
      final dirW = directionWeight(raw.box.center.dx, frameWidth);
      final (grade, ttc, score) = resolveRisk(
        classId: raw.classId,
        heightHistory: heightHistoryForRisk,
        dirW: dirW,
        boxHeight: curHeight,
        frameHeight: frameHeight,
      );

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