import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:whisper_flutter_new/whisper_flutter_new.dart';

import '../config/app_constants.dart';

/// Headless Whisper.cpp engine service.
///
/// Responsibilities are strictly scoped to the speech-to-text engine:
///  1. Unpack the bundled ggml model asset into a real file on disk
///     (whisper.cpp loads models by file path; copying by content means no
///     network download ever happens).
///  2. Own the [Whisper] engine lifecycle.
///  3. Transcribe a WAV file and return plain trimmed text.
///
/// The service is widget-independent and can be driven from the main isolate
/// or from a headless background service. Note that `rootBundle` and
/// `path_provider` require a Flutter binding in whichever isolate runs it
/// (`WidgetsFlutterBinding.ensureInitialized()`).
class WhisperService {
  Whisper? _whisper;
  bool _ready = false;

  /// Whether [initialize] has completed successfully.
  bool get isReady => _ready;

  /// Unpacks the model (if needed), creates the engine and warms it up with
  /// a native library sanity check.
  Future<void> initialize() async {
    if (_ready) return;

    final String modelDir = await _ensureModelUnpacked();

    _whisper = Whisper(
      model: WhisperModel.tiny,
      modelDir: modelDir,
    );

    // Sanity check the native library + warm up model loading.
    final String? version = await _whisper!.getVersion();
    debugPrint('whisper.cpp version: $version');

    _ready = true;
  }

  /// Copies the bundled model asset to
  /// `<documents>/<whisperModelDirName>/<whisperModelFileName>` unless a
  /// complete copy (matching [AppConstants.whisperModelBytes]) already
  /// exists. Returns the model directory path.
  Future<String> _ensureModelUnpacked() async {
    final Directory docsDir = await getApplicationDocumentsDirectory();
    final Directory modelDir =
        Directory('${docsDir.path}/${AppConstants.whisperModelDirName}');
    if (!modelDir.existsSync()) {
      modelDir.createSync(recursive: true);
    }

    final File modelFile =
        File('${modelDir.path}/${AppConstants.whisperModelFileName}');
    if (!modelFile.existsSync() ||
        modelFile.lengthSync() != AppConstants.whisperModelBytes) {
      final ByteData assetBytes =
          await rootBundle.load(AppConstants.whisperAssetPath);
      await modelFile.writeAsBytes(
        assetBytes.buffer.asUint8List(assetBytes.offsetInBytes),
        flush: true,
      );
    }
    return modelDir.path;
  }

  /// Transcribes the WAV at [wavPath] and returns the trimmed plain text.
  /// Returns an empty string when no speech was detected.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<String> transcribe(String wavPath) async {
    final Whisper? engine = _whisper;
    if (!_ready || engine == null) {
      throw StateError('WhisperService.initialize() must complete first.');
    }

    final Stopwatch sw = Stopwatch()..start();
    final WhisperTranscribeResponse response = await engine.transcribe(
      transcribeRequest: TranscribeRequest(
        audio: wavPath,
        language: AppConstants.whisperLanguage,
        isTranslate: false,
        isNoTimestamps: true, // We only want the plain text.
        threads: AppConstants.whisperThreads,
        noFallback: false,
      ),
    );
    sw.stop();
    debugPrint('Transcribed in ${sw.elapsedMilliseconds} ms');

    return response.text.trim();
  }
}
