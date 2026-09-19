import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:llamadart/llamadart.dart';
import 'package:path_provider/path_provider.dart';

import 'cleanup_utils.dart';
import 'local_text_processor.dart';

/// On-device dictation cleanup backed by the fine-tuned Qwen3.5-2B GGUF
/// ("Rambler 2B") running through llamadart (llama.cpp v0.4.0).
///
/// Falls back to the rule-based [LocalTextProcessor] whenever the model
/// cannot be loaded or generation fails/times out, so user text is never
/// lost (plan §3 graceful-fallback requirement).
class LlamaCleanupService {
  static const String _assetPath = 'assets/models/rambler-2b-q4_k_m-no-mtp.gguf';
  static const String _fileName = 'rambler-2b-q4_k_m-no-mtp.gguf';
  static const String _assetChannelName = 'mumble_jumble/assets';

  /// Exact prompt the model was fine-tuned with — do not edit here.
  static const String systemPrompt = kCleanupSystemPrompt;

  // Mirrors the training/runtime contract from plan.md: hard n_ctx cap of
  // 1024, deterministic sampling, short outputs.
  static const int _contextSize = 1024;
  static const int _maxTokens = 256;
  static const Duration _generationTimeout = Duration(seconds: 15);

  final LocalTextProcessor _fallback = LocalTextProcessor();
  final MethodChannel _assetChannel = const MethodChannel(_assetChannelName);

  LlamaEngine? _engine;
  bool _initialized = false;
  bool _generationInFlight = false;
  bool get initialized => _initialized;

  /// Loads the GGUF from app storage, copying it out of the APK asset on
  /// first launch via the Android platform channel (chunked stream copy —
  /// the 1.2 GB file must never be buffered whole in Dart heap).
  Future<void> initialize() async {
    if (_initialized) return;
    try {
      final modelPath = await _ensureModelFile();
      final engine = LlamaEngine(LlamaBackend());
      await engine.loadModel(
        modelPath,
        modelParams: const ModelParams(
          contextSize: _contextSize,
          useMmap: true,
          flashAttention: FlashAttention.auto,
        ),
      );
      _engine = engine;
      _initialized = true;
    } catch (error) {
      _engine = null;
      _initialized = false;
      debugPrint('LlamaCleanupService initialize error: $error');
    }
  }

  Future<String> _ensureModelFile() async {
    // Android: stream the bundled asset out of the APK via the platform
    // channel (rootBundle.load would buffer 1.2 GB in Dart heap).
    if (defaultTargetPlatform == TargetPlatform.android && !kIsWeb) {
      try {
        return await _assetChannel.invokeMethod<String>('ensureAssetCopied', {
          'assetPath': _assetPath,
          'fileName': _fileName,
        }) as String;
      } on PlatformException catch (error) {
        debugPrint('Asset copy channel failed: $error');
        rethrow;
      }
    }

    // Other desktop targets: try an asset-declared file, then a side-loaded
    // copy next to the app's support directory.
    final supportDir = await getApplicationSupportDirectory();
    final localFile = File('${supportDir.path}/$_fileName');
    if (await localFile.exists()) return localFile.path;

    try {
      final byteData = await rootBundle.load('assets/models/$_fileName');
      await localFile.writeAsBytes(
        byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes),
        flush: true,
      );
      return localFile.path;
    } catch (_) {
      throw StateError(
        'Rambler GGUF not found: expected $_fileName in assets or ${supportDir.path}',
      );
    }
  }

  /// Cleans a raw ASR transcript. Never throws: on any failure it returns the
  /// rule-based fallback so the user always gets usable text.
  Future<String> processText(String rawText) async {
    final input = rawText.trim();
    if (input.isEmpty) return '';

    final engine = _engine;
    if (!_initialized || engine == null || _generationInFlight) {
      return _fallback.cleanRawText(input);
    }

    _generationInFlight = true;
    try {
      final buffer = StringBuffer();
      final stream = engine
          .create(
            [
              LlamaChatMessage.fromText(
                role: LlamaChatRole.system,
                text: systemPrompt,
              ),
              LlamaChatMessage.fromText(role: LlamaChatRole.user, text: input),
            ],
            params: const GenerationParams(
              maxTokens: _maxTokens,
              temp: 0.0, // deterministic cleanup, matches greedy decoding
              topK: 0, // disable top-k filtering
              topP: 1.0, // disable nucleus filtering
              penalty: 1.0, // no repetition penalty; no-op inputs must stay stable
            ),
            enableThinking: false,
          )
          .timeout(_generationTimeout, onTimeout: (sink) {
        sink.close();
        engine.cancelGeneration();
      });

      await for (final chunk in stream) {
        final text = chunk.choices.first.delta.content;
        if (text != null) buffer.write(text);
      }

      final cleaned = postProcessCleanupOutput(buffer.toString());
      return cleaned.isNotEmpty ? cleaned : _fallback.cleanRawText(input);
    } catch (error) {
      debugPrint('LlamaCleanupService processText error: $error');
      return _fallback.cleanRawText(input);
    } finally {
      _generationInFlight = false;
    }
  }

  Future<void> dispose() async {
    final engine = _engine;
    _engine = null;
    _initialized = false;
    if (engine != null) {
      try {
        await engine.dispose();
      } catch (error) {
        debugPrint('LlamaCleanupService dispose error: $error');
      }
    }
  }
}
