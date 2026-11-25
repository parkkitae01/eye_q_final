import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:vibration/vibration.dart';

import 'object_tracker.dart';
import 'risk_engine.dart';

// ════════════════════════════════════════════════════════════════
// 수정 이력
//  [2026-11-12] 알림 과다(42초에 10회) 완화
//   1) 확인 단계 추가: 최근 1초 안에 같은 물체가 여러 프레임 잡혀야 첫 경고
//      (한두 프레임짜리 오탐으로는 울리지 않음)
//   2) 깜빡임 대응: 위협이 사라졌다고 보는 시간을 1초 → 2.5초로 늘림
//      (인식이 잠깐 끊겼다 다시 잡혀도 "새 위협"으로 치지 않음)
//   3) 같은 물체 반복 억제: 같은 종류의 "주의" 경고는 10초에 한 번만
//      (위험 경고는 기존 5초 간격 유지)
//   4) 박스 크기 기록을 물체 종류가 바뀌면 비움 (서로 다른 물체 크기가 섞이던 문제)
//   5) 계단 경고를 "위험!"(화면 80% 이상)에서 "주의"(화면 55% 이상)로 변경
//  [2026-11-24] 보행 안내 강화 — "그냥 탐지"가 아니라 "보조"임을 드러냄
//   6) 음성 안내에 방향(시계)·근사 거리·행동 지시 추가
//      예) "11시 방향 자동차, 약 3미터, 멈추세요"
//   7) 박스 크기로 대략적인 거리(m)를 추정 (단안 카메라 기반 추정값)
//  [2026-11-24] 거리 추정 다듬기
//   8) 보정 계수(kDistanceScale): 실제 측정으로 추정 거리를 맞춤
//      (앱이 3m라 할 때 실제 ~1.25m였던 관찰값을 반영)
//   9) 거리 평균(최근 프레임 중앙값): 숫자가 튀지 않게 안정화
//   10) 잘린 박스 가드: 화면 위로 잘리거나 너무 크면 미터 대신 "가까이/바로 앞"
//   11) 거리 발음: 소수점을 한글 "점"으로 (예: "약 일점오 미터")
//  [2026-11-24] 보조 기능 추가 — "탐지"가 아니라 "보행 안내"로
//   12) 우회 방향 안내: 장애물 반대쪽으로 피하도록 ("오른쪽으로 피하세요")
//   13) 안전 확인 안내: 한동안 위협이 없으면 가끔 "앞 장애물 없음"
//   14) 시작/토글 음성: 앱 시작·경고 on/off를 음성으로 알림(화면 안 봐도 됨)
// ════════════════════════════════════════════════════════════════

enum AlertLevel { none, caution, danger }

class _HSample {
  final int t; // 시각(ms)
  final double h; // 박스 높이 / 화면 높이
  _HSample(this.t, this.h);
}

class _Seen {
  final int t; // 시각(ms)
  final String label; // 그 프레임의 가장 위험한 물체 종류
  _Seen(this.t, this.label);
}

class AlertManager {
  // ── 조정 가능한 값 (현장 테스트하면서 바꾸세요) ──
  static const double kDangerTtc = 2.0; // 충돌까지 2초 이하 → 위험
  static const double kCautionTtc = 4.0; // 충돌까지 4초 이하 → 주의
  static const int kDangerCooldownMs = 5000; // 위험 경고 반복 간격
  static const int kCautionCooldownMs = 6000; // 주의 경고 반복 간격
  static const int kMinGapMs = 800; // 어떤 경고든 최소 간격
  static const int kClearMs = 2500; // 위협이 이만큼 사라지면 "새 위협"으로 취급 (기존 1000)

  // ── [추가] 확인 단계: 잠깐 스친 오탐으로는 울리지 않게 ──
  static const int kConfirmWindowMs = 1000; // 이 시간 안에
  static const int kConfirmHits = 3; // 주의: 같은 물체가 3프레임 이상 잡혀야 경고
  static const int kConfirmHitsDanger = 2; // 위험: 2프레임 이상 (빨리 알려야 하므로 더 짧게)

  // ── [추가] 같은 물체 반복 억제 ──
  static const int kSameLabelMs = 10000; // 같은 종류의 "주의" 경고는 이 간격에 한 번만

  // ── "가까이 있을 뿐, 다가오지 않음" 판단 ──
  static const int kSettleMs = 3000; // 위협 상태가 이만큼 이어지면 안정 후보
  static const int kStableWindowMs = 1500; // 이 시간 동안 박스 크기를 비교
  static const double kStableRatio = 0.08; // 크기 변화가 8% 미만이면 "안 변함"
  static const int kSettledRepeatMs = 15000; // 안정 상태에서 차분한 안내 반복 간격

  // ── [2026-11-24] 안전 확인 안내 ──
  static const int kClearAfterMs = 3000; // 위협이 사라지고 이만큼 지나야 "안전" 안내 시작
  static const int kClearAnnounceMs = 8000; // "앞 장애물 없음" 반복 간격

  // ── [2026-11-24] 근사 거리 추정 (단안 카메라 · 박스 크기 기반, 추정값) ──
  //   d ≈ 실제높이 / (2 · (박스높이/화면높이) · tan(수직화각/2))
  //   정확한 측정이 아니라 "대략 몇 미터"를 안내용으로 추정하는 값입니다.
  //   (물체가 화면에 세로로 온전히 보일 때 가장 정확하고, 비스듬하거나
  //    일부만 보이면 오차가 커집니다. 기기·대상에 맞춰 아래 값을 보정하세요.)
  static const double kVerticalFovDeg = 65.0; // 세로로 든 폰의 대략적 수직 화각(도)
  static final double _tanHalfFov =
  math.tan((kVerticalFovDeg / 2) * math.pi / 180);

  // ── 보정 계수 (가장 중요한 튜닝 값) ──
  //   공식으로 나온 추정 거리에 이 값을 곱해 실제와 맞춥니다.
  //   관찰: 앱이 "3m"라 할 때 실제 약 1.25m → 1.25 / 3 ≈ 0.42
  //   맞추는 법: 사람을 (머리~발 다 보이게) 알려진 거리에 세우고 음성을 들은 뒤
  //     새 값 = 지금 값 × (실제 거리 ÷ 앱이 말한 거리)
  static const double kDistanceScale = 0.42;

  // 거리 평균에 쓸 최근 구간(ms)
  static const int kDistWindowMs = 800;
  // 박스가 화면 위로 잘렸다고 볼 가장자리 여유(px)
  static const double kEdgePx = 3.0;
  // 박스가 이 비율 이상이면 "바로 앞"으로 (거리 계산 생략)
  static const double kVeryCloseRatio = 0.85;

  // 대상별 실제 높이(m) — 거리 추정의 기준값
  static const Map<String, double> _realHeightM = {
    'person': 1.7,
    'bicycle': 1.1,
    'car': 1.5,
    'motorcycle': 1.1,
    'bus': 3.0,
    'truck': 3.0,
    'dog': 0.5,
    'cat': 0.3,
    'fire hydrant': 0.8,
    'bench': 0.9,
    'chair': 0.9,
    'stairs': 1.2,
  };

  final FlutterTts _tts = FlutterTts();
  bool _hasVibrator = false;
  bool _ready = false;

  /// 화면을 탭하면 켜고 끌 수 있음 (시험할 때 소리가 거슬리면 끄기)
  bool enabled = true;

  // ── 상태 ──
  AlertLevel _lastLevel = AlertLevel.none;
  int _lastAlertMs = 0;
  int _lastThreatMs = 0;
  int _threatSinceMs = 0; // 위협 상태가 끊김 없이 시작된 시각 (0 = 없음)
  bool _settled = false; // 지금 "안정(가까이 있을 뿐)" 상태인가
  final List<_HSample> _hist = []; // 가장 위험한 물체의 박스 높이 기록
  String _histLabel = ''; // _hist가 어떤 물체의 기록인지

  // [추가] 확인 단계용: 최근 프레임에서 본 위협 기록
  final List<_Seen> _seen = [];

  // [추가] 마지막으로 경고한 물체 (같은 물체 반복 억제용)
  String _lastLabel = '';
  AlertLevel _lastLabelLevel = AlertLevel.none;
  int _lastLabelMs = 0;

  // [2026-11-24] 안전 확인 안내 마지막 시각
  int _lastClearAnnounceMs = 0;

  // ── 화면 표시/측정용 ──
  int lastLatencyMs = 0; // 프레임 도착 → 경고 명령까지
  String lastText = ''; // 마지막으로 말한 문장
  int alertCount = 0;
  bool get settled => _settled;

  Future<void> init() async {
    try {
      // 1) 가능하면 구글 TTS 엔진 사용 (삼성 기본 엔진보다 자연스러움)
      try {
        await _tts.setEngine('com.google.android.tts');
        debugPrint('[EYE-Q][alert] engine -> com.google.android.tts');
      } catch (e) {
        debugPrint('[EYE-Q][alert] setEngine 실패(기본 엔진 사용): $e');
      }

      final r = await _tts.setLanguage('ko-KR');
      debugPrint('[EYE-Q][alert] setLanguage(ko-KR) -> $r');

      // 2) 한국어 목소리 중 자연스러운(네트워크/뉴럴) 음성을 골라 설정
      await _pickNaturalKoVoice();

      await _tts.setSpeechRate(0.5); // 조금 느리게 (또박또박)
      await _tts.setVolume(1.0);
      await _tts.setPitch(1.05); // 살짝 높여 덜 기계적으로
    } catch (e) {
      debugPrint('[EYE-Q][alert] TTS init error: $e');
    }
    try {
      _hasVibrator = (await Vibration.hasVibrator()) == true;
    } catch (e) {
      debugPrint('[EYE-Q][alert] vibrator check error: $e');
    }
    _ready = true;
    debugPrint('[EYE-Q][alert] ready (vibrator=$_hasVibrator)');
  }

  /// 앱 시작 시 음성 안내 (화면을 못 봐도 켜진 걸 알 수 있게)
  void announceStart() {
    _lastClearAnnounceMs = DateTime.now().millisecondsSinceEpoch;
    try {
      _tts.speak('보행 보조를 시작합니다');
    } catch (_) {}
    lastText = '보행 보조를 시작합니다';
  }

  /// 경고 on/off를 음성으로 알림 (enabled 값을 바꾼 뒤 호출)
  void announceToggle() {
    try {
      _tts.stop();
      _tts.speak(enabled ? '경고 켜짐' : '경고 꺼짐');
    } catch (_) {}
    lastText = enabled ? '경고 켜짐' : '경고 꺼짐';
  }

  /// 한국어 목소리 중 자연스러운(네트워크/뉴럴) 음성을 골라 설정.
  /// (네트워크 음성은 인터넷이 있을 때 가장 자연스럽고, 없으면 기본 음성 유지)
  Future<void> _pickNaturalKoVoice() async {
    try {
      final dynamic raw = await _tts.getVoices;
      if (raw is! List) return;
      final List<Map<String, String>> ko = [];
      for (final v in raw) {
        if (v is Map) {
          final String name = '${v['name'] ?? ''}';
          final String locale = '${v['locale'] ?? ''}';
          if (locale.toLowerCase().startsWith('ko')) {
            ko.add({'name': name, 'locale': locale});
          }
        }
      }
      if (ko.isEmpty) return;

      // 네트워크/뉴럴 음성 우선 → 없으면 local 아닌 것 → 그래도 없으면 첫 번째
      Map<String, String>? pick;
      for (final v in ko) {
        final String n = v['name']!.toLowerCase();
        if (n.contains('network') ||
            n.contains('neural') ||
            n.contains('wavenet')) {
          pick = v;
          break;
        }
      }
      pick ??= ko.firstWhere(
            (v) => !v['name']!.toLowerCase().contains('local'),
        orElse: () => ko.first,
      );

      await _tts.setVoice({'name': pick['name']!, 'locale': pick['locale']!});
      debugPrint('[EYE-Q][alert] voice -> ${pick['name']} (${pick['locale']})');
    } catch (e) {
      debugPrint('[EYE-Q][alert] 음성 선택 실패(기본 음성 사용): $e');
    }
  }

  // 코코(COCO) 영어 이름 → 말하기 좋은 한국어
  static const Map<String, String> _ko = {
    'person': '사람',
    'bicycle': '자전거',
    'car': '자동차',
    'motorcycle': '오토바이',
    'bus': '버스',
    'truck': '트럭',
    'dog': '강아지',
    'cat': '고양이',
    'chair': '의자',
    'bench': '벤치',
    'couch': '소파',
    'potted plant': '화분',
    'dining table': '탁자',
    'fire hydrant': '소화전',
    'stop sign': '표지판',
    'traffic light': '신호등',
    'suitcase': '가방',
    'backpack': '가방',
    'handbag': '가방',
    'laptop': '노트북',
    'stairs': '계단',
  };

  String _koName(String label) => _ko[label] ?? '물체';

  String _direction(double boxLeft, double boxRight, double imageW) {
    if (imageW <= 0) return '앞';
    final double cx = (boxLeft + boxRight) / 2 / imageW;
    if (cx < 0.35) return '왼쪽';
    if (cx > 0.65) return '오른쪽';
    return '앞';
  }

  /// 행동 지시: 장애물 반대쪽으로 피하도록 안내.
  ///  - 계단 등 지형은 옆으로 피하는 게 아니라 속도만 조절
  ///  - 위험: 왼쪽 장애물 → 오른쪽으로, 가운데 → 멈춤
  ///  - 주의: 반대쪽으로 천천히
  String _action(AlertLevel level, Detection d, double cx) {
    if (d.classId == kStairsClassId) return '천천히 가세요';
    if (level == AlertLevel.danger) {
      if (cx < 0.4) return '오른쪽으로 피하세요';
      if (cx > 0.6) return '왼쪽으로 피하세요';
      return '멈추세요';
    }
    if (cx < 0.4) return '오른쪽으로 천천히';
    if (cx > 0.6) return '왼쪽으로 천천히';
    return '천천히 가세요';
  }

  /// 박스 중심 x로 방향을 시계 방향으로 안내.
  /// (시각장애인 보행 훈련에서 쓰는 clock-position 표기)
  String _clock(double boxLeft, double boxRight, double imageW) {
    if (imageW <= 0) return '정면';
    final double cx = (boxLeft + boxRight) / 2 / imageW;
    if (cx < 0.2) return '10시 방향';
    if (cx < 0.4) return '11시 방향';
    if (cx < 0.6) return '12시 방향';
    if (cx < 0.8) return '1시 방향';
    return '2시 방향';
  }

  /// 박스 높이 비율(박스높이/화면높이)로 대략적인 거리(m)를 추정.
  /// 대상 종류를 모르거나 계산이 불가능하면 null. (보정 계수까지 반영)
  double? _estimateMeters(String label, double heightRatio) {
    final double? hr = _realHeightM[label];
    if (hr == null || heightRatio <= 0 || _tanHalfFov <= 0) return null;
    final double d = hr / (2 * heightRatio * _tanHalfFov) * kDistanceScale;
    if (d.isNaN || d.isInfinite) return null;
    return d;
  }

  /// 최근 kDistWindowMs 동안의 박스 높이 비율 중앙값. (거리 떨림 완화)
  /// 기록이 없으면 이번 프레임 값(fallback)을 그대로 쓴다.
  double _smoothRatio(int now, double fallback) {
    final List<double> vals = [];
    for (final s in _hist) {
      if (now - s.t <= kDistWindowMs) vals.add(s.h);
    }
    if (vals.isEmpty) return fallback;
    vals.sort();
    return vals[vals.length ~/ 2]; // 중앙값(이상치에 강함)
  }

  /// 거리를 0.5m 단위로 반올림. (1.0~5.0 범위, 그 밖은 null)
  double? _roundDist(double? d) {
    if (d == null || d < 1.0 || d >= 5.0) return null;
    return (d * 2).round() / 2;
  }

  /// 화면 표시용 거리 문구 — 숫자로. (예: "약 2미터", "약 1.5미터")
  String _distDisplay(double? d) {
    if (d == null) return '';
    if (d < 1.0) return '바로 앞';
    if (d >= 5.0) return '5미터 이상';
    final double r = _roundDist(d)!;
    final String s =
    r == r.roundToDouble() ? r.toInt().toString() : r.toStringAsFixed(1);
    return '약 $s미터';
  }

  /// 음성 발음용 거리 문구 — 한글로. (TTS가 "1.5"를 "일오"로 읽는 문제 회피)
  /// 예: 1.5 → "약 일점오 미터", 2 → "약 이 미터"
  String _distSpoken(double? d) {
    if (d == null) return '';
    if (d < 1.0) return '바로 앞';
    if (d >= 5.0) return '5미터 이상';
    final double r = _roundDist(d)!;
    final int whole = r.floor();
    final bool half = (r - whole) >= 0.5;
    const List<String> ko = ['영', '일', '이', '삼', '사', '오'];
    final String wholeKo =
    (whole >= 0 && whole < ko.length) ? ko[whole] : whole.toString();
    final String num = half ? '$wholeKo점오' : wholeKo;
    return '약 $num 미터';
  }

  /// 위험 단계 결정
  ///  0순위: 계단은 박스 크기 등급(HIGH)으로만 판단
  ///  1순위: 위험 등급 이름(근접 안전장치로 강제 CRITICAL 된 경우 포함)
  ///  2순위: TTC(충돌까지 남은 시간) 기준
  AlertLevel levelOf(Detection d) {
    // [2026-11-12] 계단: 가까우면(HIGH, 박스 높이가 화면의 55% 이상) "주의"로 알림.
    //   "위험!"은 쓰지 않는다. MEDIUM/LOW(멀리 있음)는 알리지 않는다.
    if (d.classId == kStairsClassId) {
      return d.grade == RiskGrade.high ? AlertLevel.caution : AlertLevel.none;
    }
    final String g = '${kGradeName[d.grade]}'.toUpperCase();
    if (g.contains('CRIT') || g.contains('DANGER') || g.contains('위험')) {
      return AlertLevel.danger;
    }
    final double ttc = d.ttc;
    if (ttc.isFinite && ttc > 0) {
      if (ttc <= kDangerTtc) return AlertLevel.danger;
      if (ttc <= kCautionTtc) return AlertLevel.caution;
    }
    if (g.contains('WARN') || g.contains('CAUTION') || g.contains('주의')) {
      return AlertLevel.caution;
    }
    return AlertLevel.none;
  }

  /// 최근 kStableWindowMs 동안 박스 크기가 거의 안 변했는가?
  /// (기록이 충분히 쌓이기 전에는 false = "모르겠음 → 평소처럼 경고")
  bool _isSizeStable(int now) {
    if (_hist.length < 3) return false;
    final _HSample newest = _hist.last;
    _HSample? oldest;
    for (final s in _hist) {
      if (now - s.t <= kStableWindowMs) {
        oldest = s;
        break;
      }
    }
    if (oldest == null) return false;
    if (newest.t - oldest.t < kStableWindowMs * 0.8) return false;
    final double base = oldest.h < 0.01 ? 0.01 : oldest.h;
    return ((newest.h - oldest.h).abs() / base) < kStableRatio;
  }

  /// 프레임마다 호출.
  ///  [imageW]/[imageH] = 분석 이미지 크기, [frameStartMs] = 프레임이 앱에 도착한 시각(ms)
  void update(Detection? top, double imageW, double imageH, int frameStartMs) {
    if (!_ready || !enabled) return;
    final int now = DateTime.now().millisecondsSinceEpoch;

    // [추가] 확인 기록 중 오래된 것 정리
    while (_seen.isNotEmpty && now - _seen.first.t > kConfirmWindowMs) {
      _seen.removeAt(0);
    }

    // 박스 크기 기록 (가장 위험한 물체 기준)
    if (top != null && imageH > 0) {
      // [추가] 물체 종류가 바뀌면 이전 물체의 크기 기록을 버린다
      if (top.label != _histLabel) {
        _hist.clear();
        _histLabel = top.label;
      }
      final double h =
          (top.box.bottom.toDouble() - top.box.top.toDouble()) / imageH;
      _hist.add(_HSample(now, h));
    }
    while (_hist.isNotEmpty && now - _hist.first.t > kStableWindowMs + 500) {
      _hist.removeAt(0);
    }

    final AlertLevel level = top == null ? AlertLevel.none : levelOf(top);

    if (top == null || level == AlertLevel.none) {
      // 위협이 잠깐 안 보이는 건 흔하므로, 일정 시간 지나야 "해소"로 본다
      if (now - _lastThreatMs > kClearMs) {
        _lastLevel = AlertLevel.none;
        _threatSinceMs = 0;
        _settled = false;
      }
      // [2026-11-24] 안전 확인 안내: 한동안 위협·경고가 없으면 가끔 "앞 장애물 없음"
      if (now - _lastThreatMs > kClearAfterMs &&
          now - _lastAlertMs > kClearAfterMs &&
          now - _lastClearAnnounceMs >= kClearAnnounceMs) {
        _lastClearAnnounceMs = now;
        try {
          _tts.speak('앞 장애물 없음');
        } catch (_) {}
        lastText = '앞 장애물 없음';
      }
      return;
    }

    // ── [추가] 확인 단계 ──
    // 최근 kConfirmWindowMs 안에 같은 종류가 몇 프레임 잡혔는지 센다.
    // 연속일 필요는 없다 (인식이 한 프레임 끊겨도 다시 세지 않도록).
    _seen.add(_Seen(now, top.label));
    int hits = 0;
    for (final s in _seen) {
      if (s.label == top.label) hits++;
    }
    final int need =
    level == AlertLevel.danger ? kConfirmHitsDanger : kConfirmHits;
    if (hits < need) return; // 아직 확인 안 됨 → 위협으로 치지 않는다

    _lastThreatMs = now;
    if (_threatSinceMs == 0) _threatSinceMs = now;

    // ── 안정(가까이 있을 뿐) 판단 ──
    final bool wasSettled = _settled;
    _settled = (now - _threatSinceMs >= kSettleMs) && _isSizeStable(now);
    if (wasSettled && !_settled) {
      // 다시 커지기 시작 → 새 위협처럼 즉시 경고
      _lastLevel = AlertLevel.none;
    }

    // ── [추가] 방금 경고한 것과 같은 물체인가? ──
    // 같은 종류이고, 단계가 더 높아지지 않았고, 아직 kSameLabelMs가 안 지났으면 true
    final bool sameAsLast = top.label == _lastLabel &&
        level.index <= _lastLabelLevel.index &&
        (now - _lastLabelMs) < kSameLabelMs;

    final int sinceLast = now - _lastAlertMs;
    bool fire;
    if (_settled) {
      fire = sinceLast >= kSettledRepeatMs;
    } else {
      bool escalated = level.index > _lastLevel.index;
      // 깜빡였다 다시 잡힌 같은 물체는 "새 위협"으로 치지 않는다
      if (escalated && sameAsLast) escalated = false;
      final int cooldown =
      level == AlertLevel.danger ? kDangerCooldownMs : kCautionCooldownMs;
      fire = (escalated && sinceLast >= kMinGapMs) ||
          (!escalated && sinceLast >= cooldown);
      // 같은 물체의 "주의" 경고는 kSameLabelMs에 한 번만 (위험 경고는 그대로)
      if (fire && sameAsLast && level != AlertLevel.danger) fire = false;
    }
    if (!fire) return;

    _lastLevel = level;
    _lastAlertMs = now;
    _lastLabel = top.label;
    _lastLabelLevel = level;
    _lastLabelMs = now;
    _emit(level, top, imageW, imageH, frameStartMs, calm: _settled);
  }

  void _emit(AlertLevel level, Detection d, double imageW, double imageH,
      int frameStartMs,
      {required bool calm}) {
    final int now = DateTime.now().millisecondsSinceEpoch;
    final String name = _koName(d.label);
    final String dir =
    _clock(d.box.left.toDouble(), d.box.right.toDouble(), imageW);

    // 이번 프레임 박스 높이 비율
    final double curRatio = imageH > 0
        ? (d.box.bottom.toDouble() - d.box.top.toDouble()) / imageH
        : 0.0;
    // 거리 문구: 화면 표시용(숫자)과 음성 발음용(한글)을 따로 만든다.
    //   (잘린 박스 가드 → 평균 비율 → 보정 거리)
    final String distDisplay; // 예: "약 2미터"  (화면)
    final String distSpoken; // 예: "약 이 미터" (음성)
    if (curRatio >= kVeryCloseRatio) {
      distDisplay = distSpoken = '바로 앞';
    } else if (d.box.top.toDouble() <= kEdgePx) {
      distDisplay = distSpoken = '가까이';
    } else {
      final double sRatio = _smoothRatio(now, curRatio);
      final double? m = _estimateMeters(d.label, sRatio);
      distDisplay = _distDisplay(m);
      distSpoken = _distSpoken(m);
    }

    // 안내 문장: 방향 + 대상 + (거리) + 행동(우회 안내)
    //   textDisplay = 화면 카드용, textSpoken = TTS용
    final double cx = imageW > 0
        ? (d.box.left.toDouble() + d.box.right.toDouble()) / 2 / imageW
        : 0.5;
    final String textDisplay;
    final String textSpoken;
    if (calm) {
      textDisplay = distDisplay.isEmpty
          ? '$dir $name, 가까이 있음'
          : '$dir $name, $distDisplay, 가까이 있음';
      textSpoken = distSpoken.isEmpty
          ? '$dir $name, 가까이 있음'
          : '$dir $name, $distSpoken, 가까이 있음';
    } else {
      final String action = _action(level, d, cx);
      final List<String> disp = ['$dir $name'];
      final List<String> spok = ['$dir $name'];
      if (distDisplay.isNotEmpty) {
        disp.add(distDisplay);
        spok.add(distSpoken);
      }
      disp.add(action);
      spok.add(action);
      textDisplay = disp.join(', ');
      textSpoken = spok.join(', ');
    }

    // 진동 (차분한 안내는 아주 짧게)
    if (_hasVibrator) {
      try {
        if (calm) {
          Vibration.vibrate(duration: 80);
        } else if (level == AlertLevel.danger) {
          Vibration.vibrate(pattern: [0, 250, 100, 250, 100, 500]);
        } else {
          Vibration.vibrate(duration: 150);
        }
      } catch (e) {
        debugPrint('[EYE-Q][alert] vibrate error: $e');
      }
    }

    // 음성 (위험은 말하던 걸 끊고 바로) — 발음용 문장을 읽는다
    try {
      if (!calm && level == AlertLevel.danger) {
        _tts.stop();
      }
      _tts.speak(textSpoken);
    } catch (e) {
      debugPrint('[EYE-Q][alert] speak error: $e');
    }

    // 지연 기록: 프레임 도착 → 여기(경고 명령 직후)
    lastLatencyMs = DateTime.now().millisecondsSinceEpoch - frameStartMs;
    lastText = textDisplay; // 화면 카드에는 숫자 문구를 표시
    alertCount++;
    debugPrint('[EYE-Q][alert] #$alertCount $textDisplay '
        '${calm ? "(calm) " : ""}'
        'ttc=${d.ttc.isFinite ? d.ttc.toStringAsFixed(1) : "--"} '
        'latency=${lastLatencyMs}ms');
  }

  void dispose() {
    try {
      _tts.stop();
      Vibration.cancel();
    } catch (_) {}
  }
}