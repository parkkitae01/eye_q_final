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
// 최근 프레임 기록 기반 TTC (초)  — 노이즈에 강한 버전
//   - history: (누적시간, 박스높이) 기록, 오래된 것→최신 순
//   - 최근 최대 5개 프레임의 추세선(최소제곱법 기울기)으로 성장 속도를 구함
//   - 기록이 2개 미만이거나, 추세가 "안 커짐"이면 → 무한대
// ─────────────────────────────────────────────
double estimateTtc(List<(double, double)> history) {
  if (history.length < 2) return double.infinity;

  final window =
  history.length > 5 ? history.sublist(history.length - 5) : history;

  final n = window.length;
  final tMean = window.map((p) => p.$1).reduce((a, b) => a + b) / n;
  final hMean = window.map((p) => p.$2).reduce((a, b) => a + b) / n;

  double num = 0, den = 0;
  for (final (t, h) in window) {
    num += (t - tMean) * (h - hMean);
    den += (t - tMean) * (t - tMean);
  }
  if (den == 0) return double.infinity;

  final growthRate = num / den; // 초당 픽셀 증가량 (추세선 기울기)
  if (growthRate <= 0) return double.infinity;

  final curHeight = window.last.$2;
  return curHeight / growthRate;
}

// ─────────────────────────────────────────────
// 근접 안전장치: 박스 높이가 화면 높이의 80% 이상이면
// TTC/트랙 히스토리 상태와 무관하게 무조건 CRITICAL로 강제 처리
//   - 근접 상태에서는 트랙 ID 리셋으로 TTC가 신뢰 불가능해지므로
//     "거의 충돌 임박"에 해당하는 작은 TTC(0.1초)로 강제 대입
// ─────────────────────────────────────────────
const double kProximityCriticalRatio = 0.8;
const double kForcedProximityTtc = 0.1;

// ─────────────────────────────────────────────
// 계단: 다가오는 물체가 아니라 고정 지형 → TTC 대신 박스 크기(=거리)로 판단
//   (현장 테스트하면서 바꾸세요. 아래 두 값은 임시 기준입니다)
// ─────────────────────────────────────────────
const int kStairsClassId = 80; // 계단 모델 전용 번호 (사람=0과 겹침 방지)
const double kStairsMediumRatio = 0.30; // 박스 높이/화면 높이 이 이상 → MEDIUM
const double kStairsHighRatio = 0.55; // 이 이상 → HIGH (계단은 80% 이상이어도 HIGH, CRITICAL 없음)

bool isProximityCritical(double boxHeight, double frameHeight) {
  if (frameHeight <= 0) return false;
  return (boxHeight / frameHeight) >= kProximityCriticalRatio;
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
// 최종 위험 판정 (grade, ttc, score)을 한 번에 계산.
//   - 기본 흐름: getRiskLevel() → estimateTtc() → computeRiskScore()
//   - 계단(kStairsClassId)은 TTC 대신 박스 크기(거리)로 등급을 매김
//     (근접 안전장치보다 먼저 판단 → 계단은 CRITICAL이 되지 않음)
//   - 그 외 물체는 근접 안전장치(isProximityCritical)에 걸리면
//     classId나 트랙 히스토리 상태와 무관하게 CRITICAL로 덮어씀
//   Detection을 만드는 호출부(트래커/메인 루프)에서는
//   getRiskLevel()/estimateTtc()/computeRiskScore()를 따로 부르지 말고
//   이 함수 하나만 호출하면 됨.
// ─────────────────────────────────────────────
(RiskGrade, double, double) resolveRisk({
  required int classId,
  required List<(double, double)> heightHistory,
  required double dirW,
  required double boxHeight,
  required double frameHeight,
}) {
  // 계단: TTC가 아니라 박스 크기(거리)로 등급 결정
  //  [2026-11-12] 근접 안전장치보다 먼저 판단한다.
  //   계단은 다가오는 물체가 아닌 고정 지형이므로 강제 CRITICAL(TTC 0.1초)을 적용하지 않음.
  //   (화면을 크게 덮는 오탐 박스가 "위험!" 경고로 나가던 문제 수정)
  if (classId == kStairsClassId) {
    final ratio = frameHeight > 0 ? boxHeight / frameHeight : 0.0;
    final g = ratio >= kStairsHighRatio
        ? RiskGrade.high
        : (ratio >= kStairsMediumRatio ? RiskGrade.medium : RiskGrade.low);
    return (g, double.infinity, kGradeWeight[g]! * ratio * dirW);
  }

  if (isProximityCritical(boxHeight, frameHeight)) {
    final score = computeRiskScore(
      RiskGrade.critical,
      kForcedProximityTtc,
      dirW,
    );
    return (RiskGrade.critical, kForcedProximityTtc, score);
  }

  final grade = getRiskLevel(classId);
  final ttc = estimateTtc(heightHistory);
  final score = computeRiskScore(grade, ttc, dirW);
  return (grade, ttc, score);
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