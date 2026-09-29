// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : 경량 객체 트래커 (블록 4-2)
// ════════════════════════════════════════════════════════════════
// Python model.track()이 주던 객체 ID를 IoU 매칭으로 직접 부여하고,
// 최근 프레임 높이 기록(추세선)으로 TTC·위험도 점수까지 계산한다.
// 파일 위치: lib/object_tracker.dart
//
// ⚠️ 수정: 클래스(라벨) 오인식 대응
//   - 기존엔 "같은 클래스일 때만" IoU 매칭 → classifier가 한 프레임만
//     흔들려도(예: dog→cat) 새 트랙이 생기면서 라벨이 그대로 튀고 TTC 리셋됨
//   - 변경: 1차(같은 클래스+IoU) → 실패 시 2차(클래스 무시+IoU)로 매칭해서
//     트랙 연속성을 유지하고, 화면에 보여줄 클래스는 트랙별 최근 5개 다수결로 확정
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
  Rect box;
  final List<(double, double)> heightHistory; // (누적시간, 박스높이) 최근 5개
  final List<(int, String)> classHistory;     // (classId, label) 최근 5개 — 다수결용
  int missed; // 연속으로 매칭 안 된 프레임 수

  _Track(this.id, int classId, String label, this.box, double height, double t, this.missed)
      : heightHistory = [(t, height)],
        classHistory = [(classId, label)];

  void addHeight(double t, double height) {
    heightHistory.add((t, height));
    if (heightHistory.length > 5) heightHistory.removeAt(0);
  }

  void addClass(int classId, String label) {
    classHistory.add((classId, label));
    if (classHistory.length > 5) classHistory.removeAt(0);
  }

  // ── 최근 기록 중 가장 많이 나온 클래스 (다수결) ──
  //    동점이면 더 최근에 나온 쪽을 우선 (뒤에서부터 세기)
  (int, String) get majorityClass {
    final counts = <int, int>{};
    final labelOf = <int, String>{};
    for (final (cid, lbl) in classHistory) {
      counts[cid] = (counts[cid] ?? 0) + 1;
      labelOf[cid] = lbl;
    }
    int bestId = classHistory.last.$1;
    int bestCount = 0;
    for (final (cid, cnt) in counts.entries.map((e) => (e.key, e.value))) {
      if (cnt > bestCount) {
        bestCount = cnt;
        bestId = cid;
      }
    }
    return (bestId, labelOf[bestId]!);
  }

  int get majorityClassId => majorityClass.$1;
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
    final matched = List<int?>.filled(raws.length, null); // raw 인덱스 → 매칭된 트랙 인덱스

    // ── 1차 매칭: 같은 클래스(다수결 기준) + IoU ──
    for (int ri = 0; ri < raws.length; ri++) {
      final raw = raws[ri];
      double bestIou = iouThreshold;
      int bestIdx = -1;
      for (int i = 0; i < _tracks.length; i++) {
        if (used.contains(i)) continue;
        if (_tracks[i].majorityClassId != raw.classId) continue;
        final iou = _iou(_tracks[i].box, raw.box);
        if (iou >= bestIou) {
          bestIou = iou;
          bestIdx = i;
        }
      }
      if (bestIdx >= 0) {
        used.add(bestIdx);
        matched[ri] = bestIdx;
      }
    }

    // ── 2차 매칭(fallback): 1차에서 못 찾은 것만, 클래스 무시하고 IoU로만 ──
    //    classifier가 그 프레임만 순간적으로 다른 클래스로 착각한 경우를 구제
    for (int ri = 0; ri < raws.length; ri++) {
      if (matched[ri] != null) continue;
      final raw = raws[ri];
      double bestIou = iouThreshold;
      int bestIdx = -1;
      for (int i = 0; i < _tracks.length; i++) {
        if (used.contains(i)) continue;
        final iou = _iou(_tracks[i].box, raw.box);
        if (iou >= bestIou) {
          bestIou = iou;
          bestIdx = i;
        }
      }
      if (bestIdx >= 0) {
        used.add(bestIdx);
        matched[ri] = bestIdx;
      }
    }

    // ── 결과 조립 ──
    for (int ri = 0; ri < raws.length; ri++) {
      final raw = raws[ri];
      final curHeight = raw.box.height;
      final matchedIdx = matched[ri];

      int trackId;
      List<(double, double)> heightHistoryForRisk;
      int dispClassId;
      String dispLabel;

      if (matchedIdx != null) {
        // 기존 물체와 매칭됨 → ID 승계 + 기록 추가
        final tr = _tracks[matchedIdx];
        trackId = tr.id;
        tr.addHeight(t, curHeight);
        tr.addClass(raw.classId, raw.label); // 클래스도 기록에 추가 (다수결용)
        heightHistoryForRisk = tr.heightHistory;
        tr.box = raw.box;
        tr.missed = 0;

        final (mid, mlabel) = tr.majorityClass;
        dispClassId = mid;
        dispLabel = mlabel;
      } else {
        // 처음 본 물체 → 새 ID 발급 (기록 1개뿐이니 TTC는 resolveRisk 내부에서 무한대 처리)
        trackId = _nextId++;
        final newTrack =
        _Track(trackId, raw.classId, raw.label, raw.box, curHeight, t, 0);
        newTracks.add(newTrack);
        heightHistoryForRisk = newTrack.heightHistory;
        dispClassId = raw.classId;
        dispLabel = raw.label;
      }

      // 위험도 계산 (risk_engine.dart의 두뇌 사용)
      //   - 다수결로 확정된 클래스(dispClassId)를 기준으로 판단 (raw 그대로 X)
      final dirW = directionWeight(raw.box.center.dx, frameWidth);
      final (grade, ttc, score) = resolveRisk(
        classId: dispClassId,
        heightHistory: heightHistoryForRisk,
        dirW: dirW,
        boxHeight: curHeight,
        frameHeight: frameHeight,
      );

      results.add(Detection(
        box: raw.box,
        trackId: trackId,
        classId: dispClassId,
        label: dispLabel,
        grade: grade,
        ttc: ttc,
        score: score,
      ));
    }

    // 이번 프레임에 안 잡힌 기존 트랙 → missed 증가, 오래되면 삭제
    for (int i = 0; i < _tracks.length; i++) {
      if (!used.contains(i)) _tracks[i].missed++;
    }
    _tracks.removeWhere((tr) => tr.missed > maxMissed);
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