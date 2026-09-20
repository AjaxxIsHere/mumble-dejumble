import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import '../config/app_constants.dart';

/// Headless audio capture service wrapping the `record` package.
///
/// Knows nothing about Flutter widgets, Whisper, or LLMs: it can record a
/// whisper.cpp-compatible WAV file from the main isolate or be injected into
/// a headless background service unchanged.
class AudioRecorderService {
  AudioRecorderService({AudioRecorder? recorder})
      : _recorder = recorder ?? AudioRecorder();

  final AudioRecorder _recorder;

  /// Requests the RECORD_AUDIO runtime permission if it has not been
  /// granted yet, and reports whether recording may proceed.
  Future<bool> ensurePermission() => _recorder.hasPermission();

  /// Starts recording a whisper-compatible WAV and returns the file path
  /// the audio will be written to.
  ///
  /// Any stale recording from a previous session is deleted first so we
  /// never transcribe old audio.
  Future<String> startRecording() async {
    final Directory docsDir = await getApplicationDocumentsDirectory();
    final String wavPath =
        '${docsDir.path}/${AppConstants.recordingFileName}';

    final File oldFile = File(wavPath);
    if (oldFile.existsSync()) oldFile.deleteSync();

    await _recorder.start(AppConstants.whisperRecordConfig, path: wavPath);
    return wavPath;
  }

  /// Stops recording and returns the path of the completed WAV, or null if
  /// the platform did not report one.
  Future<String?> stopRecording() => _recorder.stop();

  /// Releases the underlying platform recorder.
  Future<void> dispose() => _recorder.dispose();
}
