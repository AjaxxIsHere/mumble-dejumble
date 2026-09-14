import 'package:shared_preferences/shared_preferences.dart';

class SettingsService {
  static const String _showRawTranscriptKey = 'show_raw_transcript';
  static const String _autoCopyResultKey = 'auto_copy_result';

  SharedPreferences? _prefs;

  Future<void> initialize() async {
    _prefs ??= await SharedPreferences.getInstance();
  }

  Future<bool> get showRawTranscript async {
    await initialize();
    return _prefs?.getBool(_showRawTranscriptKey) ?? true;
  }

  Future<void> setShowRawTranscript(bool value) async {
    await initialize();
    await _prefs?.setBool(_showRawTranscriptKey, value);
  }

  Future<bool> get autoCopyResult async {
    await initialize();
    return _prefs?.getBool(_autoCopyResultKey) ?? false;
  }

  Future<void> setAutoCopyResult(bool value) async {
    await initialize();
    await _prefs?.setBool(_autoCopyResultKey, value);
  }
}
