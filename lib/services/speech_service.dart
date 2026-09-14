import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';

class SpeechService {
  final AudioRecorder _recorder = AudioRecorder();
  bool _isRecording = false;

  bool get isRecording => _isRecording;

  Future<void> initialize() async {
    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      throw StateError('Microphone permission was not granted.');
    }
  }

  Future<String> startRecording() async {
    if (kIsWeb) {
      throw UnsupportedError('WAV recording is not supported on web in this V1 build.');
    }

    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      throw StateError('Microphone permission was not granted.');
    }

    if (_isRecording) {
      throw StateError('Recording is already active.');
    }

    final tempDir = Directory.systemTemp.createTempSync('mumble_jumble');
    final filePath = '${tempDir.path}/capture_${DateTime.now().millisecondsSinceEpoch}.wav';

    await _recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
        autoGain: true,
        echoCancel: true,
        noiseSuppress: true,
      ),
      path: filePath,
    );

    _isRecording = true;
    return filePath;
  }

  Future<String> stopRecording() async {
    if (!_isRecording) {
      return '';
    }

    try {
      final path = await _recorder.stop();
      if (path != null && path.isNotEmpty) {
        return path;
      }
      return '';
    } finally {
      _isRecording = false;
    }
  }

  Future<void> dispose() async {
    if (_isRecording) {
      await stopRecording();
    }
    await _recorder.cancel();
  }
}
