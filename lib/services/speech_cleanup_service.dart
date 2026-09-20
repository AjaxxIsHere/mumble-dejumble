import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:llamadart/llamadart.dart';

import '../config/app_constants.dart';

/// Headless LLM service that cleans raw Whisper transcripts using the
/// bundled Rambler GGUF model via LlamaDart.
///
/// Guarantees:
///  * The model is loaded lazily on first use and reused afterwards.
///  * Requests are serialized through an in-memory task queue, so two
///    overlapping [clean] calls can never run concurrent inference against
///    the same engine.
///  * The engine is never handed to callers; only cleaned text is returned.
class SpeechCleanupService {
  LlamaEngine? _engine;
  Future<void>? _loading;
  Future<void>? _inference;

  /// Whether the GGUF model has been loaded.
  bool get isReady => _engine != null;

  /// Loads the model on first use and reuses it for subsequent requests.
  Future<void> initialize() {
    return _loading ??= _loadModel();
  }

  Future<void> _loadModel() async {
    // The GGUF cannot be memory-mapped out of the asset bundle, so ask the
    // native side to stream it into app-private storage first.
    final String modelPath =
        await const MethodChannel(AppConstants.assetsChannel)
                .invokeMethod<String>(
              AppConstants.ensureAssetCopiedMethod,
              <String, String>{
                AppConstants.assetPathArg: AppConstants.cleanupModelAssetPath,
                AppConstants.fileNameArg: AppConstants.cleanupModelFileName,
              },
            ) ??
        (throw StateError('Model asset copy returned no path.'));

    final engine = LlamaEngine(LlamaBackend());
    await engine.loadModel(
      modelPath,
      modelParams: const ModelParams(
        contextSize: AppConstants.cleanupContextSize,
        numberOfThreads: AppConstants.cleanupThreads,
        numberOfThreadsBatch: AppConstants.cleanupThreads,
        loadMtp: false,
      ),
    );
    _engine = engine;
    debugPrint('Rambler 2B cleanup model loaded');
  }

  /// Cleans one Whisper transcript without retaining conversation history.
  ///
  /// Falls back to the raw text when the model produces nothing usable.
  Future<String> clean(String rawText) {
    Future<String> request() async {
      await initialize();
      final engine = _engine!;
      final messages = [
        LlamaChatMessage.fromText(
          role: LlamaChatRole.system,
          text: AppConstants.cleanupSystemPrompt,
        ),
        LlamaChatMessage.fromText(
          role: LlamaChatRole.user,
          text: '${AppConstants.noThinkPrefix}$rawText',
        ),
      ];
      final output = StringBuffer();

      await for (final chunk in engine.create(
        messages,
        params: const GenerationParams(
          maxTokens: AppConstants.cleanupMaxTokens,
          temp: AppConstants.cleanupTemperature,
          topK: AppConstants.cleanupTopK,
          topP: AppConstants.cleanupTopP,
          penalty: AppConstants.cleanupPenalty,
          stopSequences: AppConstants.cleanupStopSequences,
        ),
        enableThinking: false,
      )) {
        if (chunk.choices.isEmpty) continue;
        final text = chunk.choices.first.delta.content;
        if (text != null) output.write(text);
      }

      // Strip any think block / chat-template tokens that leaked through.
      final cleaned = output
          .toString()
          .replaceAll(RegExp(r'<think>[\s\S]*?</think>'), '')
          .replaceAll(RegExp(r'<\|im_end\|>'), '')
          .trim();
      return cleaned.isEmpty ? rawText.trim() : cleaned;
    }

    // Task serialization: chain this request after the previous one and keep
    // a non-throwing tail as the new queue head so a failed inference never
    // poisons later requests.
    final previous = _inference ?? Future<void>.value();
    final result = previous.then((_) => request());
    _inference = result.then<void>((_) {}, onError: (_, _) {});
    return result;
  }

  /// Releases the native LLM engine. Safe to call more than once.
  Future<void> dispose() async {
    final engine = _engine;
    _engine = null;
    _loading = null;
    if (engine != null) await engine.dispose();
  }
}
