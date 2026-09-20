import 'package:flutter/foundation.dart';

/// High-level phases of the dictation pipeline.
///
/// These values replace the competing boolean flags of the previous
/// implementation (`_isRecording`, `_isTranscribing`, `_modelReady`) so the
/// UI can only ever observe one coherent activity at a time.
enum DictationStatus {
  /// Whisper model is being unpacked from assets and loaded by the engine.
  initializing,

  /// Engine ready and waiting for the user to start dictating.
  idle,

  /// Capturing microphone audio.
  recording,

  /// Running whisper.cpp on the captured WAV.
  transcribing,

  /// Running the LLM cleanup pass on the raw transcript.
  cleaning,

  /// Model initialization failed; dictation stays unavailable until the
  /// app is restarted. Transient runtime errors do NOT use this status —
  /// they are reported via [DictationState.errorMessage] while the pipeline
  /// returns to [idle].
  failed,
}

/// Immutable snapshot of everything the UI needs to render the dictation
/// experience. The [DictationController] replaces this object wholesale and
/// notifies listeners; widgets never mutate application state.
@immutable
class DictationState {
  const DictationState({
    this.status = DictationStatus.initializing,
    this.rawTranscript = '',
    this.cleanedTranscript = '',
    this.errorMessage,
    this.overlayPermissionGranted = false,
  });

  /// State at app start: model loading, no transcripts, no permission data.
  const DictationState.initial() : this();

  /// Current phase of the dictation pipeline.
  final DictationStatus status;

  /// Raw whisper.cpp output ('' until the first transcription completes).
  final String rawTranscript;

  /// LLM-cleaned output ('' until the first cleanup completes).
  final String cleanedTranscript;

  /// Last user-facing error, if any. Cleared when the next action starts.
  final String? errorMessage;

  /// Whether the "Display over other apps" permission has been granted.
  final bool overlayPermissionGranted;

  /// True once the Whisper engine is loaded and usable.
  bool get isModelReady =>
      status != DictationStatus.initializing && status != DictationStatus.failed;

  /// True while the pipeline is capturing, transcribing or cleaning.
  bool get isBusy =>
      status == DictationStatus.recording ||
      status == DictationStatus.transcribing ||
      status == DictationStatus.cleaning;

  /// Whether the mic button should accept taps (start or stop recording).
  bool get canToggleRecording =>
      status == DictationStatus.idle || status == DictationStatus.recording;

  /// Returns a copy with the given fields replaced.
  ///
  /// [clearError] is used instead of passing `errorMessage: null` so that
  /// "keep the existing error" and "remove the error" are distinguishable.
  DictationState copyWith({
    DictationStatus? status,
    String? rawTranscript,
    String? cleanedTranscript,
    String? errorMessage,
    bool? overlayPermissionGranted,
    bool clearError = false,
  }) {
    return DictationState(
      status: status ?? this.status,
      rawTranscript: rawTranscript ?? this.rawTranscript,
      cleanedTranscript: cleanedTranscript ?? this.cleanedTranscript,
      errorMessage: clearError ? null : (errorMessage ?? this.errorMessage),
      overlayPermissionGranted:
          overlayPermissionGranted ?? this.overlayPermissionGranted,
    );
  }

  @override
  String toString() =>
      'DictationState(status: $status, raw: "${rawTranscript.length} chars", '
      'cleaned: "${cleanedTranscript.length} chars", '
      'error: $errorMessage, overlayPermission: $overlayPermissionGranted)';
}
