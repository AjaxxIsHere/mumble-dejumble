import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:llamadart/llamadart.dart';

/// Runs the Rambler GGUF model used to clean Whisper transcripts.
class SpeechCleanupModel {
  static const String _assetPath =
      'assets/models/rambler-2b-q4_k_m-no-mtp.gguf';
  static const String _modelFileName = 'rambler-2b-q4_k_m-no-mtp.gguf';
  static const MethodChannel _assetChannel = MethodChannel(
    'mumble_jumble/assets',
  );

static const String _systemPrompt =
    'Clean up this voice dictation: remove filler words, stutters, and '
    'self-corrections; fix punctuation and capitalization. Keep every fact, '
    'name, date, time, and number exactly as dictated. Output only the '
    'cleaned text. Do not reason aloud. Do not output a think block.';

  LlamaEngine? _engine;
  Future<void>? _loading;
  Future<void>? _inference;

  /// Loads the model on first use and reuses it for subsequent requests.
  Future<void> initialize() {
    return _loading ??= _loadModel();
  }

  Future<void> _loadModel() async {
    final String modelPath =
        await _assetChannel.invokeMethod<String>(
          'ensureAssetCopied',
          <String, String>{'assetPath': _assetPath, 'fileName': _modelFileName},
        ) ??
        (throw StateError('Model asset copy returned no path.'));

    final engine = LlamaEngine(LlamaBackend());
    await engine.loadModel(
      modelPath,
      modelParams: const ModelParams(
        contextSize: 2048,
        numberOfThreads: 4,
        numberOfThreadsBatch: 4,
        loadMtp: false,
      ),
    );
    _engine = engine;
    debugPrint('Rambler 2B cleanup model loaded');
  }

  /// Cleans one Whisper transcript without retaining conversation history.
  Future<String> clean(String rawText) {
    Future<String> request() async {
      await initialize();
      final engine = _engine!;
      final messages = [
        LlamaChatMessage.fromText(
          role: LlamaChatRole.system,
          text: _systemPrompt,
        ),
        LlamaChatMessage.fromText(
          role: LlamaChatRole.user,
          text: '/no_think\n$rawText',
        ),
      ];
      final output = StringBuffer();

      await for (final chunk in engine.create(
        messages,
        params: const GenerationParams(
          maxTokens: 128,
          temp: 0.0,
          topK: 1,
          topP: 1.0,
          penalty: 1.0,
          stopSequences: ['<|im_end|>', '<|im_start|>'],
        ),
        enableThinking: false,
      )) {
        if (chunk.choices.isEmpty) continue;
        final text = chunk.choices.first.delta.content;
        if (text != null) output.write(text);
      }

      final cleaned = output
          .toString()
          .replaceAll(RegExp(r'<think>[\s\S]*?</think>'), '')
          .replaceAll(RegExp(r'<\|im_end\|>'), '')
          .trim();
      return cleaned.isEmpty ? rawText.trim() : cleaned;
    }

    final previous = _inference ?? Future<void>.value();
    final result = previous.then((_) => request());
    _inference = result.then<void>((_) {}, onError: (_, _) {});
    return result;
  }

  Future<void> dispose() async {
    final engine = _engine;
    _engine = null;
    _loading = null;
    if (engine != null) await engine.dispose();
  }
}
