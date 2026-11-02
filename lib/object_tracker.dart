// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : 경량 객체 트래커 (블록 4-2)
// ════════════════════════════════════════════════════════════════
// Python model.track()이 주던 객체 ID를 IoU 매칭으로 직접 부여하고,
// 최근 프레임 높이 기록(추세선)으로 TTC·위험도 점수까지 계산한다.
// 파일 위치: lib/object_tracker.dart
//
// ⚠️ 수정 1: 클래스(라벨) 오인식 대응
//   - 1차(같은 클래스+IoU) → 실패 시 2차(클래스 무시+IoU)로 매칭해서
//     트랙 연속성을 유지하고, 화면에 보여줄 클래스는 최근 5개 다수결로 확정
//
// ⚠️ 수정 2: 박스 깜빡임 대응 (놓친 프레임에서도 박스 잠시 유지)
//   - 계단처럼 검출 신뢰도가 경계선에 걸리는 물체는 "잡혔다 놓쳤다"를 반복함
//   - 2프레임 이상 연속 확인된 트랙이 갑자기 안 잡히면, 최대 holdFrames 프레임 동안
//     마지막 결과를 그대로 내보내서 박스·경고가 끊기지 않게 함
//   - 한 번만 잡힌 오탐(1프레임짜리)은 유지하지 않음 (hits < 2)
//   - 실제로 이번 프레임에 잡힌 것만 보고 싶으면 realDetections 사용
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
  final int classId;  // 클래스 번호
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
  int hits = 1; // 지금까지 실제로 잡힌 횟수 (1프레임짜리 오탐 구분용)
  Detection? last; // 가장 최근 "실제 탐지"로 만든 결과 (놓친 프레임에서 유지용)

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
  final int holdFrames;      // 놓친 뒤에도 박스를 유지하는 최대 프레임 수
  final int minHitsToHold;   // 이 횟수 이상 잡힌 트랙만 유지 대상

  int _nextId = 0;
  final List<_Track> _tracks = [];

  List<Detection> _lastReal = const [];

  /// 이번 프레임에 "실제로 탐지된" 결과만 (유지된 박스 제외).
  /// 경고(음성·진동)를 실제 탐지 기준으로만 울리고 싶을 때 사용.
  List<Detection> get realDetections => _lastReal;

  ObjectTracker({
    this.iouThreshold = 0.3,
    this.maxMissed = 5,
    this.holdFrames = 4,
    this.minHitsToHold = 2,
  });

  // ⚠️ t = 추적 시작 시점부터 누적된 시간(초). 프레임 간 간격(delta)이 아님.
  // ⚠️ frameHeight: 근접 안전장치(resolveRisk)가 "박스 높이가 화면의 80% 이상인지"
  //    판단하려면 화면 세로 길이가 필요함.
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

      _Track curTrack;
      int trackId;
      List<(double, double)> heightHistoryForRisk;
      int dispClassId;
      String dispLabel;

      if (matchedIdx != null) {
        // 기존 물체와 매칭됨 → ID 승계 + 기록 추가
        final tr = _tracks[matchedIdx];
        curTrack = tr;
        trackId = tr.id;
        tr.addHeight(t, curHeight);
        tr.addClass(raw.classId, raw.label);
        heightHistoryForRisk = tr.heightHistory;
        tr.box = raw.box;
        tr.missed = 0;
        tr.hits++;

        final (mid, mlabel) = tr.majorityClass;
        dispClassId = mid;
        dispLabel = mlabel;
      } else {
        // 처음 본 물체 → 새 ID 발급
        trackId = _nextId++;
        final newTrack =
        _Track(trackId, raw.classId, raw.label, raw.box, curHeight, t, 0);
        newTracks.add(newTrack);
        curTrack = newTrack;
        heightHistoryForRisk = newTrack.heightHistory;
        dispClassId = raw.classId;
        dispLabel = raw.label;
      }

      // 위험도 계산 (risk_engine.dart의 두뇌 사용)
      final dirW = directionWeight(raw.box.center.dx, frameWidth);
      final (grade, ttc, score) = resolveRisk(
        classId: dispClassId,
        heightHistory: heightHistoryForRisk,
        dirW: dirW,
        boxHeight: curHeight,
        frameHeight: frameHeight,
      );

      final det = Detection(
        box: raw.box,
        trackId: trackId,
        classId: dispClassId,
        label: dispLabel,
        grade: grade,
        ttc: ttc,
        score: score,
      );
      curTrack.last = det; // 다음 프레임에 놓치면 이 결과를 잠시 유지
      results.add(det);
    }

    // 이번 프레임에 안 잡힌 기존 트랙 → missed 증가
    for (int i = 0; i < _tracks.length; i++) {
      if (!used.contains(i)) _tracks[i].missed++;
    }

    // "실제 탐지"만 따로 보관 (유지 박스를 넣기 전에 복사)
    _lastReal = List<Detection>.of(results);

    // ── 놓친 프레임에서도 박스 잠시 유지 (깜빡임 완화) ──
    //    조건: 1~holdFrames 프레임째 놓침 + 이전에 minHitsToHold번 이상 확인된 트랙
    for (final tr in _tracks) {
      if (tr.missed >= 1 &&
          tr.missed <= holdFrames &&
          tr.hits >= minHitsToHold &&
          tr.last != null) {
        results.add(tr.last!);
      }
    }

    // 오래 안 보인 트랙 삭제, 새 물체는 다음 프레임부터 추적 대상
    _tracks.removeWhere((tr) => tr.missed > maxMissed);
    _tracks.addAll(newTracks);

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