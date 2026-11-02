import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:vibration/vibration.dart';

import 'object_tracker.dart';
import 'risk_engine.dart';

enum AlertLevel { none, caution, danger }

class _HSample {
  final int t; // 시각(ms)
  final double h; // 박스 높이 / 화면 높이
  _HSample(this.t, this.h);
}

class AlertManager {
  // ── 조정 가능한 값 (현장 테스트하면서 바꾸세요) ──
  static const double kDangerTtc = 2.0; // 충돌까지 2초 이하 → 위험
  static const double kCautionTtc = 4.0; // 충돌까지 4초 이하 → 주의
  static const int kDangerCooldownMs = 5000; // 위험 경고 반복 간격
  static const int kCautionCooldownMs = 6000; // 주의 경고 반복 간격
  static const int kMinGapMs = 800; // 어떤 경고든 최소 간격
  static const int kClearMs = 1000; // 위협이 이만큼 사라지면 "새 위협"으로 취급

  // ── "가까이 있을 뿐, 다가오지 않음" 판단 ──
  static const int kSettleMs = 3000; // 위협 상태가 이만큼 이어지면 안정 후보
  static const int kStableWindowMs = 1500; // 이 시간 동안 박스 크기를 비교
  static const double kStableRatio = 0.08; // 크기 변화가 8% 미만이면 "안 변함"
  static const int kSettledRepeatMs = 15000; // 안정 상태에서 차분한 안내 반복 간격

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

  // ── 화면 표시/측정용 ──
  int lastLatencyMs = 0; // 프레임 도착 → 경고 명령까지
  String lastText = ''; // 마지막으로 말한 문장
  int alertCount = 0;
  bool get settled => _settled;

  Future<void> init() async {
    try {
      final r = await _tts.setLanguage('ko-KR');
      debugPrint('[EYE-Q][alert] setLanguage(ko-KR) -> $r');
      await _tts.setSpeechRate(0.55);
      await _tts.setVolume(1.0);
      await _tts.setPitch(1.0);
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

  /// 위험 단계 결정
  ///  1순위: 위험 등급 이름(근접 안전장치로 강제 CRITICAL 된 경우 포함)
  ///  2순위: TTC(충돌까지 남은 시간) 기준
  AlertLevel levelOf(Detection d) {
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

    // 박스 크기 기록 (가장 위험한 물체 기준)
    if (top != null && imageH > 0) {
      final double h =
          (top.box.bottom.toDouble() - top.box.top.toDouble()) / imageH;
      _hist.add(_HSample(now, h));
    }
    while (_hist.isNotEmpty && now - _hist.first.t > kStableWindowMs + 500) {
      _hist.removeAt(0);
    }

    final AlertLevel level = top == null ? AlertLevel.none : levelOf(top);

    if (level == AlertLevel.none) {
      // 위협이 잠깐 안 보이는 건 흔하므로, 일정 시간 지나야 "해소"로 본다
      if (now - _lastThreatMs > kClearMs) {
        _lastLevel = AlertLevel.none;
        _threatSinceMs = 0;
        _settled = false;
      }
      return;
    }
    _lastThreatMs = now;
    if (_threatSinceMs == 0) _threatSinceMs = now;

    // ── 안정(가까이 있을 뿐) 판단 ──
    final bool wasSettled = _settled;
    _settled = (now - _threatSinceMs >= kSettleMs) && _isSizeStable(now);
    if (wasSettled && !_settled) {
      // 다시 커지기 시작 → 새 위협처럼 즉시 경고
      _lastLevel = AlertLevel.none;
    }

    final int sinceLast = now - _lastAlertMs;
    bool fire;
    if (_settled) {
      fire = sinceLast >= kSettledRepeatMs;
    } else {
      final bool escalated = level.index > _lastLevel.index;
      final int cooldown =
      level == AlertLevel.danger ? kDangerCooldownMs : kCautionCooldownMs;
      fire = (escalated && sinceLast >= kMinGapMs) ||
          (!escalated && sinceLast >= cooldown);
    }
    if (!fire || top == null) return;

    _lastLevel = level;
    _lastAlertMs = now;
    _emit(level, top, imageW, frameStartMs, calm: _settled);
  }

  void _emit(AlertLevel level, Detection d, double imageW, int frameStartMs,
      {required bool calm}) {
    final String name = _koName(d.label);
    final String dir =
    _direction(d.box.left.toDouble(), d.box.right.toDouble(), imageW);
    final String text = calm
        ? '$dir $name, 가까이 있음'
        : (level == AlertLevel.danger ? '위험! $dir $name' : '주의, $dir $name');

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

    // 음성 (위험은 말하던 걸 끊고 바로)
    try {
      if (!calm && level == AlertLevel.danger) {
        _tts.stop();
      }
      _tts.speak(text);
    } catch (e) {
      debugPrint('[EYE-Q][alert] speak error: $e');
    }

    // 지연 기록: 프레임 도착 → 여기(경고 명령 직후)
    lastLatencyMs = DateTime.now().millisecondsSinceEpoch - frameStartMs;
    lastText = text;
    alertCount++;
    debugPrint('[EYE-Q][alert] #$alertCount $text '
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