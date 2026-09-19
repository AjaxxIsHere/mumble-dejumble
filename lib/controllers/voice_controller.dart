import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:whisper_flutter_new/whisper_flutter_new.dart';

import '../models/app_state.dart';
import '../services/llama_cleanup_service.dart';
import '../services/local_text_processor.dart';
import '../services/model_manager.dart';
import '../services/permission_service.dart';
import '../services/speech_service.dart';

class VoiceController extends ChangeNotifier {
  final PermissionService _permissionService = PermissionService();
  final ModelManager _modelManager = ModelManager();
  final LlamaCleanupService _cleanupService = LlamaCleanupService();
  final LocalTextProcessor _textProcessor = LocalTextProcessor();
  final SpeechService _speechService = SpeechService();

  VoiceAppState _state = VoiceAppState.idle;
  String _errorMessage = '';
  String _rawTranscript = '';
  String _polishedText = '';
  bool _showRawTranscript = true;
  bool _autoCopy = false;
  bool _isBusy = false;
  bool _isRecording = false;

  VoiceAppState get state => _state;
  String get errorMessage => _errorMessage;
  String get rawTranscript => _rawTranscript;
  String get polishedText => _polishedText;
  bool get showRawTranscript => _showRawTranscript;
  bool get autoCopy => _autoCopy;
  bool get isBusy => _isBusy;
  bool get isRecording => _isRecording;
  bool get sttReady => _modelManager.sttInitialized;
  bool get llmReady => _modelManager.llmInitialized;

  /// True when the fine-tuned GGUF engine loaded; false means cleanup
  /// currently runs through the rule-based fallback.
  bool get llmEngineReady => _cleanupService.initialized;

  Future<void> initialize() async {
    _state = VoiceAppState.initializingStt;
    notifyListeners();

    try {
      final hasPermission = await _permissionService.isMicrophoneGranted;
      if (!hasPermission) {
        _state = VoiceAppState.ready;
        _errorMessage = 'Microphone permission needed to use local voice typing.';
        notifyListeners();
        return;
      }

      await _modelManager.initializeStt();
      await _prepareLocalTinyModel();
      await _cleanupService.initialize();
      await _modelManager.initializeLlm();
      _state = VoiceAppState.ready;
      _errorMessage = '';
      notifyListeners();
    } catch (error) {
      _state = VoiceAppState.error;
      _errorMessage = 'The local models could not start.';
      debugPrint('VoiceController initialize error: $error');
      notifyListeners();
    }
  }

  Future<void> requestPermissions() async {
    final granted = await _permissionService.requestMicrophonePermission();
    if (!granted) {
      _state = VoiceAppState.error;
      _errorMessage = 'Could not access the microphone. Please grant permission in Settings.';
      notifyListeners();
      return;
    }

    _state = VoiceAppState.ready;
    notifyListeners();
  }

  Future<String> _prepareLocalTinyModel() async {
    final supportDir = await getApplicationSupportDirectory();
    final destination = File('${supportDir.path}/ggml-tiny.bin');

    if (!await destination.exists()) {
      final bytes = await rootBundle.load('assets/models/ggml-tiny.en.bin');
      await destination.writeAsBytes(
        bytes.buffer.asUint8List(bytes.offsetInBytes, bytes.lengthInBytes),
        flush: true,
      );
    }

    return destination.path;
  }

  Future<void> startListening() async {
    if (_isBusy || _isRecording) return;

    _isBusy = true;
    _state = VoiceAppState.recording;
    _errorMessage = '';
    _rawTranscript = '';
    _polishedText = '';
    notifyListeners();

    try {
      final hasPermission = await _permissionService.isMicrophoneGranted;
      if (!hasPermission) {
        throw StateError('Microphone permission was not granted.');
      }

      await _speechService.startRecording();
      _isRecording = true;
      _state = VoiceAppState.recording;
      notifyListeners();
    } catch (error) {
      _isBusy = false;
      _isRecording = false;
      _state = VoiceAppState.error;
      _errorMessage = 'Voice capture failed. Please try again.';
      debugPrint('VoiceController startListening error: $error');
      notifyListeners();
    }
  }

  Future<void> stopListening() async {
    if (!_isRecording) return;

    _state = VoiceAppState.processing;
    notifyListeners();

    try {
      final audioPath = await _speechService.stopRecording();
      _rawTranscript = await _transcribeAudio(audioPath);
      final cleaned = await _cleanupService.processText(_rawTranscript);
      _polishedText = cleaned.isNotEmpty ? cleaned : _textProcessor.cleanRawText(_rawTranscript);
      _state = VoiceAppState.completed;
      _isRecording = false;
      _isBusy = false;
      notifyListeners();
    } catch (error) {
      _state = VoiceAppState.error;
      _errorMessage = 'Transcription failed. Please try again.';
      _isBusy = false;
      _isRecording = false;
      debugPrint('VoiceController stopListening error: $error');
      notifyListeners();
    }
  }

  Future<String> _transcribeAudio(String audioPath) async {
    final modelDir = await getApplicationSupportDirectory();
    final whisper = Whisper(
      model: WhisperModel.tiny,
      modelDir: modelDir.path,
    );

    final response = await whisper.transcribe(
      transcribeRequest: TranscribeRequest(
        audio: audioPath,
        language: 'en',
        isNoTimestamps: true,
        splitOnWord: false,
      ),
    );

    final text = response.text.trim();
    return text;
  }

  Future<void> copyResult() async {
    if (_polishedText.isEmpty) return;
    await Clipboard.setData(ClipboardData(text: _polishedText));
    _state = VoiceAppState.completed;
    notifyListeners();
  }

  void setShowRawTranscript(bool value) {
    _showRawTranscript = value;
    notifyListeners();
  }

  void setAutoCopy(bool value) {
    _autoCopy = value;
    notifyListeners();
  }

  @override
  void dispose() {
    _speechService.dispose();
    _cleanupService.dispose();
    super.dispose();
  }
}
