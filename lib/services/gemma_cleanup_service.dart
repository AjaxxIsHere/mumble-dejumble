import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_gemma/flutter_gemma.dart';

import 'local_text_processor.dart';

class GemmaCleanupService {
  final LocalTextProcessor _fallback = LocalTextProcessor();
  bool _initialized = false;
  bool _assetAvailable = false;
  InferenceModelSession? _session;

  bool get initialized => _initialized;
  bool get assetAvailable => _assetAvailable;

  Future<void> initialize() async {
    try {
      final manifest = await rootBundle.loadString('AssetManifest.json');
      final hasAsset = manifest.contains('assets/models/gemma-3-270m-it.task') ||
          manifest.contains('assets/models/gemma-3-270m-it.bin') ||
          manifest.contains('assets/models/gemma-3-270m-it.litertlm');

      if (!hasAsset) {
        _assetAvailable = false;
        _initialized = false;
        return;
      }

      await FlutterGemma.initialize();
      await FlutterGemma.installModel(
        modelType: ModelType.gemmaIt,
        fileType: ModelFileType.task,
      ).fromAsset('models/gemma-3-270m-it.task').install();

      final model = await FlutterGemma.getActiveModel(maxTokens: 256);
      _session = await model.createSession();
      _assetAvailable = true;
      _initialized = true;
    } catch (error) {
      _assetAvailable = false;
      _initialized = false;
      debugPrint('GemmaCleanupService initialize error: $error');
    }
  }

  Future<String> processText(String rawText) async {
    final fallback = _fallback.cleanRawText(rawText);
    if (!_initialized || _session == null) {
      return fallback;
    }

    try {
      final prompt = 'Clean this spoken transcript into polished written text. Remove filler words, repetition, and fix obvious grammar. Keep names, dates, times, and numbers. Avoid inventing details. Return only the final text.\nInput: $rawText';
      await _session!.addQueryChunk(Message.text(text: prompt, isUser: true));
      final result = await _session!.getResponse();
      final finalText = result.trim();
      if (finalText.isNotEmpty) {
        return finalText;
      }
    } catch (error) {
      debugPrint('GemmaCleanupService processText error: $error');
    }

    return fallback;
  }

  Future<void> dispose() async {
    await _session?.close();
    _session = null;
    _initialized = false;
  }
}
