import 'dart:async';
import 'package:flutter/foundation.dart';

class ModelManager {
  bool _sttInitialized = false;
  bool _llmInitialized = false;
  bool _sttLoaded = false;
  bool _llmLoaded = false;
  String? _sttError;
  String? _llmError;
  double _sttProgress = 0.0;
  double _llmProgress = 0.0;
  bool _isBusy = false;

  bool get sttInitialized => _sttInitialized;
  bool get llmInitialized => _llmInitialized;
  bool get sttLoaded => _sttLoaded;
  bool get llmLoaded => _llmLoaded;
  String? get sttError => _sttError;
  String? get llmError => _llmError;
  double get sttProgress => _sttProgress;
  double get llmProgress => _llmProgress;
  bool get isBusy => _isBusy;

  Future<void> initializeStt() async {
    if (_sttInitialized) return;
    _isBusy = true;
    _sttProgress = 0.1;
    try {
      await Future<void>.delayed(const Duration(milliseconds: 200));
      _sttInitialized = true;
      _sttLoaded = true;
      _sttError = null;
      _sttProgress = 1.0;
    } catch (error) {
      _sttError = 'STT model failed to initialize.';
      debugPrint('ModelManager initializeStt error: $error');
    } finally {
      _isBusy = false;
    }
  }

  Future<void> initializeLlm() async {
    if (_llmInitialized) return;
    _isBusy = true;
    _llmProgress = 0.1;
    try {
      await Future<void>.delayed(const Duration(milliseconds: 250));
      _llmInitialized = true;
      _llmLoaded = true;
      _llmError = null;
      _llmProgress = 1.0;
    } catch (error) {
      _llmError = 'Rambler GGUF failed to initialize.';
      debugPrint('ModelManager initializeLlm error: $error');
    } finally {
      _isBusy = false;
    }
  }

  void markSttError(String message) {
    _sttError = message;
    _sttLoaded = false;
    _sttInitialized = false;
  }

  void markLlmError(String message) {
    _llmError = message;
    _llmLoaded = false;
    _llmInitialized = false;
  }
}
