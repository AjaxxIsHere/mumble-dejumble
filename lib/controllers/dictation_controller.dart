import 'dart:async';

import 'package:flutter/foundation.dart';

import '../models/dictation_state.dart';
import '../services/audio_recorder_service.dart';
import '../services/overlay_controller.dart';
import '../services/speech_cleanup_service.dart';
import '../services/whisper_service.dart';

/// Placeholder shown when whisper.cpp returns no speech at all.
const String kNoSpeechDetected = '(No speech detected)';

/// Single source of truth for the dictation feature.
///
/// Orchestrates the pipeline Audio -> Whisper -> LlamaDart and exposes one
/// immutable [DictationState] to listeners. It contains no engine code and
/// no widget code: engines are injected [AudioRecorderService],
/// [WhisperService], [SpeechCleanupService] and [OverlayController]
/// collaborators (wired in `main.dart`), which keeps this class testable and
/// headless services reusable elsewhere.
class DictationController extends ChangeNotifier {
  DictationController({
    required AudioRecorderService audioRecorderService,
    required WhisperService whisperService,
    required SpeechCleanupService speechCleanupService,
    required OverlayController overlayController,
  })  : _audio = audioRecorderService,
        _whisper = whisperService,
        _cleanup = speechCleanupService,
        _overlay = overlayController;

  final AudioRecorderService _audio;
  final WhisperService _whisper;
  final SpeechCleanupService _cleanup;
  final OverlayController _overlay;

  DictationState _state = const DictationState.initial();

  /// The current immutable state snapshot.
  DictationState get state => _state;

  bool _initialized = false;
  bool _disposed = false;

  // -------------------------------------------------------------------------
  //  State plumbing
  // -------------------------------------------------------------------------

  void _update(DictationState next) {
    if (_disposed) return;
    _state = next;
    notifyListeners();
  }

  void _mutate(DictationState Function(DictationState) mutator) =>
      _update(mutator(_state));

  // -------------------------------------------------------------------------
  //  Lifecycle
  // -------------------------------------------------------------------------

  /// Idempotent startup: pre-warms the overlay permission flow + native
  /// volume-key listener, and loads the Whisper engine in parallel.
  Future<void> initialize() async {
    if (_initialized) return;
    _initialized = true;

    unawaited(_initOverlayPermission());
    await _initWhisper();
  }

  Future<void> _initOverlayPermission() async {
    try {
      await _overlay.initialize();
      final granted = await _overlay.isPermissionGranted();
      _mutate((s) => s.copyWith(overlayPermissionGranted: granted));
    } catch (e) {
      debugPrint('Overlay init failed: $e');
    }
  }

  Future<void> _initWhisper() async {
    try {
      await _whisper.initialize();
      _mutate((s) =>
          s.copyWith(status: DictationStatus.idle, clearError: true));
    } catch (e) {
      // Initialization failures are fatal for dictation: keep the mic
      // disabled until the app restarts.
      _mutate((s) => s.copyWith(
            status: DictationStatus.failed,
            errorMessage: 'Model init failed: $e',
          ));
    }
  }

  /// Requests the "Display over other apps" permission and records the
  /// outcome in state.
  Future<void> requestOverlayPermission() async {
    try {
      final granted = await _overlay.requestPermission();
      _mutate((s) => s.copyWith(overlayPermissionGranted: granted));
    } catch (e) {
      debugPrint('Overlay permission request failed: $e');
    }
  }

  /// Deep link into the accessibility settings page for the volume trigger.
  Future<void> openAccessibilitySettings() =>
      _overlay.openAccessibilitySettings();

  // -------------------------------------------------------------------------
  //  Dictation pipeline
  // -------------------------------------------------------------------------

  /// Starts recording when idle, stops + transcribes when recording.
  /// Ignored in every other state.
  Future<void> toggleRecording() async {
    final status = _state.status;
    if (status == DictationStatus.recording) {
      await _stopAndTranscribe();
      return;
    }
    if (status != DictationStatus.idle) return;
    await _startRecording();
  }

  Future<void> _startRecording() async {
    _mutate((s) => s.copyWith(
          rawTranscript: '',
          cleanedTranscript: '',
          clearError: true,
        ));

    try {
      // ensurePermission() requests the RECORD_AUDIO runtime permission if
      // it has not been granted yet.
      if (!await _audio.ensurePermission()) {
        _mutate((s) => s.copyWith(
              errorMessage: 'Microphone permission denied.',
            ));
        return;
      }
      await _audio.startRecording();
      _mutate((s) => s.copyWith(status: DictationStatus.recording));
    } catch (e) {
      _mutate((s) => s.copyWith(
            status: DictationStatus.idle,
            errorMessage: 'Record start failed: $e',
          ));
    }
  }

  Future<void> _stopAndTranscribe() async {
    String? path;
    try {
      path = await _audio.stopRecording();
    } catch (e) {
      _mutate((s) => s.copyWith(
            status: DictationStatus.idle,
            errorMessage: 'Record stop failed: $e',
          ));
      return;
    }
    if (path == null) {
      // Make sure we never strand the UI in a busy state.
      _mutate((s) => s.copyWith(status: DictationStatus.idle));
      return;
    }

    _mutate((s) => s.copyWith(status: DictationStatus.transcribing));

    try {
      final String raw = await _whisper.transcribe(path);

      if (raw.isEmpty) {
        _mutate((s) => s.copyWith(
              status: DictationStatus.idle,
              rawTranscript: kNoSpeechDetected,
            ));
        return;
      }

      _mutate((s) => s.copyWith(
            status: DictationStatus.cleaning,
            rawTranscript: raw,
          ));

      final Stopwatch cleanupSw = Stopwatch()..start();
      final String cleaned = await _cleanup.clean(raw);
      cleanupSw.stop();
      debugPrint('Cleaned in ${cleanupSw.elapsedMilliseconds} ms');

      _mutate((s) => s.copyWith(
            status: DictationStatus.idle,
            cleanedTranscript: cleaned,
          ));
    } catch (e) {
      debugPrint('Transcription failed: $e');
      _mutate((s) => s.copyWith(
            status: DictationStatus.idle,
            errorMessage: 'Transcription failed: $e',
          ));
    }
  }

  // -------------------------------------------------------------------------
  //  Teardown
  // -------------------------------------------------------------------------

  /// Releases every owned service. The Whisper engine has no explicit
  /// dispose API (freed with the process); the overlay controller only
  /// drops its channel handlers.
  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _overlay.dispose();
    unawaited(_audio.dispose());
    unawaited(_cleanup.dispose());
    super.dispose();
  }
}
