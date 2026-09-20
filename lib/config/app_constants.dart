import 'package:flutter/painting.dart' show Color;
import 'package:record/record.dart';

/// Central configuration for Mumble (De)Jumble.
///
/// Every magic string (method channels, asset paths, model file names) and
/// every tuned numeric constant (recording parameters, inference limits,
/// overlay geometry) lives here so native code, services, and UI always
/// agree. Pure Dart and safe to import from the overlay secondary isolate.
abstract final class AppConstants {
  // -------------------------------------------------------------------------
  //  Method channels (must match MainActivity.kt / VolumeKeyAccessibilityService.kt)
  // -------------------------------------------------------------------------

  /// Channel that streams bundled model assets to app-private storage.
  /// Implemented natively in MainActivity.
  static const String assetsChannel = 'mumble_jumble/assets';

  /// Methods on [assetsChannel].
  static const String ensureAssetCopiedMethod = 'ensureAssetCopied';

  /// Channel used for two-way communication with the native side:
  ///  - native -> Dart: `volumeTrigger` ('show' | 'hide') from the
  ///    accessibility service, plus `isOverlayActive` state queries.
  ///  - Dart -> native: `openAccessibilitySettings` deep link.
  static const String overlayTriggerChannel = 'mumble_jumble/overlay_trigger';

  /// Methods on [overlayTriggerChannel].
  static const String volumeTriggerMethod = 'volumeTrigger';
  static const String isOverlayActiveMethod = 'isOverlayActive';
  static const String openAccessibilitySettingsMethod =
      'openAccessibilitySettings';

  /// Arguments for [volumeTriggerMethod].
  static const String volumeTriggerShow = 'show';
  static const String volumeTriggerHide = 'hide';

  /// Arguments for [ensureAssetCopiedMethod].
  static const String assetPathArg = 'assetPath';
  static const String fileNameArg = 'fileName';

  // -------------------------------------------------------------------------
  //  Overlay engine
  // -------------------------------------------------------------------------

  /// The cached overlay engine tag used by flutter_overlay_window internally
  /// (OverlayConstants.CACHED_TAG). Mirrored here so the accessibility
  /// service and Dart agree on it.
  static const String overlayEngineTag = 'myCachedEngine';

  /// Messages exchanged over FlutterOverlayWindow.overlayListener.
  static const String overlayHideMessage = 'hide_overlay';
  static const String overlayTapMessage = 'bubble_tap';

  // -------------------------------------------------------------------------
  //  Whisper (speech-to-text)
  // -------------------------------------------------------------------------

  /// Bundled whisper.cpp model asset (tiny.en, English-only).
  static const String whisperAssetPath = 'assets/models/ggml-tiny.en.bin';

  /// File name the plugin expects inside [whisperModelDirName].
  static const String whisperModelFileName = 'ggml-tiny.bin';

  /// Sub-directory (under the app documents directory) for whisper models.
  static const String whisperModelDirName = 'whisper';

  /// Exact byte size of the bundled tiny.en model. Used to detect truncated
  /// or corrupted copies without hashing the whole file.
  static const int whisperModelBytes = 77704715;

  /// Recording configuration tuned for whisper.cpp: the bundled native code
  /// requires 16 kHz, 16-bit PCM WAV (mono or stereo) — mono halves the
  /// data to transcribe.
  static const RecordConfig whisperRecordConfig = RecordConfig(
    encoder: AudioEncoder.wav,
    sampleRate: 16000,
    numChannels: 1,
    bitRate: 256000,
    echoCancel: false,
    noiseSuppress: false,
    autoGain: false,
    androidConfig: AndroidRecordConfig(audioSource: AndroidAudioSource.mic),
  );

  /// Transcription language: tiny.en is English-only, so 'auto' would fail.
  static const String whisperLanguage = 'en';

  /// CPU threads used by whisper.cpp inference.
  static const int whisperThreads = 4;

  /// Name of the WAV file written to the documents directory.
  static const String recordingFileName = 'recording.wav';

  // -------------------------------------------------------------------------
  //  Speech cleanup (LlamaDart / Rambler)
  // -------------------------------------------------------------------------

  /// Bundled GGUF model used to clean raw Whisper transcripts.
  static const String cleanupModelAssetPath =
      'assets/models/rambler-2b-q4_k_m-no-mtp.gguf';

  /// File name the native asset copier stores it under (inside filesDir).
  static const String cleanupModelFileName = 'rambler-2b-q4_k_m-no-mtp.gguf';

  /// System prompt for the cleanup model. Keep verbatim: downstream
  /// behaviour depends on the exact wording.
  static const String cleanupSystemPrompt =
      'Clean up this voice dictation: remove filler words, stutters, and '
      'self-corrections; fix punctuation and capitalization. Keep every fact, '
      'name, date, time, and number exactly as dictated. Output only the '
      'cleaned text. Do not reason aloud. Do not output a think block.';

  /// Prefix prepended to user text so reasoning models skip their think block.
  static const String noThinkPrefix = '/no_think\n';

  /// Stop sequences emitted by the Qwen-style chat template.
  static const List<String> cleanupStopSequences = <String>[
    '<|im_end|>',
    '<|im_start|>',
  ];

  /// Generation parameters for the cleanup pass.
  static const int cleanupMaxTokens = 512;
  static const double cleanupTemperature = 0.0;
  static const int cleanupTopK = 1;
  static const double cleanupTopP = 1.0;
  static const double cleanupPenalty = 1.0;

  /// Context and threading for the cleanup model.
  static const int cleanupContextSize = 2048;
  static const int cleanupThreads = 4;

  // -------------------------------------------------------------------------
  //  Overlay bubble geometry, timing & visuals
  // -------------------------------------------------------------------------

  /// Overlay window dimensions passed to FlutterOverlayWindow.showOverlay.
  static const int overlayWidth = 220;
  static const int overlayHeight = 96;

  /// How long the hide() handshake waits for the bubble's slide-out
  /// animation before force-closing the overlay engine. Must outlast the
  /// animation in mic_bubble.dart (260 ms).
  static const Duration overlayExitDelay = Duration(milliseconds: 300);

  /// Nothing-style bubble palette: yellow pill with a dark accent.
  static const Color overlayPillColor = Color(0xFFF7C000);
  static const Color overlayAccentColor = Color(0xFF18181B);
}
