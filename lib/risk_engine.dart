// ════════════════════════════════════════════════════════════════
// 👁️  EYE-Q : 위험 판단 두뇌 (Python test_ttc.py → Dart 이식)
// ════════════════════════════════════════════════════════════════
// 박스 변화율 기반 TTC + 위험도 점수. 공식은 Python 검증본과 동일.
// 파일 위치: lib/risk_engine.dart
// ════════════════════════════════════════════════════════════════

import 'dart:ui';

// ─────────────────────────────────────────────
// 위험 등급 (Python의 'CRITICAL'/'HIGH'/... 문자열 대신 enum)
// ─────────────────────────────────────────────
enum RiskGrade { critical, high, medium, low }

// ─────────────────────────────────────────────
// COCO class id → 위험 등급 매핑  (Python: RISK_LEVELS)
// ─────────────────────────────────────────────
const Map<int, RiskGrade> kRiskLevels = {
  // CRITICAL: 차량류 (bicycle, car, motorcycle, bus, truck)
  1: RiskGrade.critical,
  2: RiskGrade.critical,
  3: RiskGrade.critical,
  5: RiskGrade.critical,
  7: RiskGrade.critical,
  // HIGH: 사람·소화전
  0: RiskGrade.high,
  10: RiskGrade.high,
  // MEDIUM: 정보성 (traffic light, bench, chair)
  9: RiskGrade.medium,
  13: RiskGrade.medium,
  56: RiskGrade.medium,
};

// 등급별 가중치  (Python: GRADE_WEIGHT)
const Map<RiskGrade, double> kGradeWeight = {
  RiskGrade.critical: 4.0,
  RiskGrade.high: 3.0,
  RiskGrade.medium: 2.0,
  RiskGrade.low: 1.0,
};

// 등급별 색상  (Python: RISK_COLORS, BGR → Dart는 ARGB)
const Map<RiskGrade, Color> kGradeColor = {
  RiskGrade.critical: Color(0xFFFF0000), // 빨강
  RiskGrade.high: Color(0xFFFF8C00),     // 주황
  RiskGrade.medium: Color(0xFFFFFF00),   // 노랑
  RiskGrade.low: Color(0xFFB4B4B4),      // 회색
};

// 등급 → 화면 표시용 문자열 (HUD 라벨)
const Map<RiskGrade, String> kGradeName = {
  RiskGrade.critical: 'CRITICAL',
  RiskGrade.high: 'HIGH',
  RiskGrade.medium: 'MEDIUM',
  RiskGrade.low: 'LOW',
};

// ─────────────────────────────────────────────
// class id → 위험 등급 (없으면 LOW)  (Python: get_risk_level)
// ─────────────────────────────────────────────
RiskGrade getRiskLevel(int classId) => kRiskLevels[classId] ?? RiskGrade.low;

// ─────────────────────────────────────────────
// 박스 변화율 기반 TTC (초)  (Python: estimate_ttc)
//   - 이전 높이가 없거나 dt<=0  → 무한대
//   - 박스가 안 커짐(멀어짐/정지) → 무한대
// ─────────────────────────────────────────────
double estimateTtc(double? prevHeight, double curHeight, double dt) {
  if (prevHeight == null || dt <= 0) return double.infinity;
  final dh = curHeight - prevHeight;
  if (dh <= 0) return double.infinity;
  final growthRate = dh / dt;
  return curHeight / growthRate;
}

// ─────────────────────────────────────────────
// 박스 중심 X → 방향 가중치  (Python: direction_weight)
// ─────────────────────────────────────────────
double directionWeight(double cx, double frameWidth) {
  final relX = cx / frameWidth; // 0.0 ~ 1.0
  if (relX >= 0.35 && relX <= 0.65) return 1.5; // 정면 (11~1시)
  if (relX >= 0.15 && relX <= 0.85) return 1.0; // 측면
  return 0.3;                                   // 경로 밖
}

// ─────────────────────────────────────────────
// 위험도 점수 = 등급가중치 ÷ TTC × 방향가중치
//   (Python: compute_risk_score)
// ─────────────────────────────────────────────
double computeRiskScore(RiskGrade grade, double ttc, double dirW) {
  if (ttc.isInfinite || ttc <= 0) return 0;
  return (kGradeWeight[grade]! / ttc) * dirW;
}

// ─────────────────────────────────────────────
// 한 객체의 탐지 결과 그릇
//   Python의 hazards[i] 딕셔너리(box/id/label/grade/ttc/score)에 해당
// ─────────────────────────────────────────────
class Detection {
  final Rect box;        // 픽셀 좌표 박스 (left, top, right, bottom)
  final int trackId;     // 추적 ID — 다음 블록(4-2) 트래커가 채움
  final int classId;     // COCO 클래스 번호
  final String label;    // 클래스 이름 (예: 'person')
  final RiskGrade grade; // 위험 등급
  final double ttc;      // 초 (무한대면 위험 없음)
  final double score;    // 위험도 점수 (높을수록 위험)

  const Detection({
    required this.box,
    required this.trackId,
    required this.classId,
    required this.label,
    required this.grade,
    required this.ttc,
    required this.score,
  });

  // 화면 상단 요약 문구 만들기 (Python의 summary 문자열과 동일 포맷)
  String get summaryText =>
      'TOP THREAT: $label | TTC: ${ttc.isInfinite ? "--" : ttc.toStringAsFixed(1)}s'
          ' | Score: ${score.toStringAsFixed(2)}';
}