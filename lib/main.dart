import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:whisper_flutter_new/whisper_flutter_new.dart';

import 'overlay.dart';

/// Entry point used by flutter_overlay_window's secondary Flutter engine.
@pragma('vm:entry-point')
void overlayMain() {
  runOverlayApp();
}

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mumble Jumble',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const VoiceScreen(),
    );
  }
}

/// Recording config tuned for whisper.cpp:
/// the bundled native code requires 16 kHz, 16-bit PCM WAV (mono or stereo).
const RecordConfig _whisperRecordConfig = RecordConfig(
  encoder: AudioEncoder.wav,
  sampleRate: 16000,
  numChannels: 1,
  bitRate: 256000,
  echoCancel: false,
  noiseSuppress: false,
  autoGain: false,
  androidConfig: AndroidRecordConfig(
    audioSource: AndroidAudioSource.mic,
  ),
);

class VoiceScreen extends StatefulWidget {
  const VoiceScreen({super.key});

  @override
  State<VoiceScreen> createState() => _VoiceScreenState();
}

class _VoiceScreenState extends State<VoiceScreen> {
  final AudioRecorder _recorder = AudioRecorder();
  final OverlayController _overlayController = OverlayController();

  /// True once the user granted "Display over other apps".
  bool _overlayPermissionReady = false;

  /// Whisper engine. [modelDir] points at the directory that will contain
  /// our bundled model copied under the name the plugin expects
  /// ("ggml-tiny.bin"). Copying by content means no network download happens.
  Whisper? _whisper;

  bool _isRecording = false;
  bool _isTranscribing = false;
  bool _modelReady = false;
  String? _error;
  String _transcription = '';

  @override
  void initState() {
    super.initState();
    _initWhisper();
    _initOverlay();
  }

  Future<void> _initOverlay() async {
    // Pre-warm the overlay permission flow and start listening for native
    // volume-key triggers as early as possible.
    await _overlayController.initialize();
    final granted = await _overlayController.isPermissionGranted();
    if (mounted) {
      setState(() => _overlayPermissionReady = granted);
    }
  }

  Future<void> _requestOverlayPermission() async {
    final granted = await _overlayController.requestPermission();
    if (mounted) {
      setState(() => _overlayPermissionReady = granted);
    }
  }

  @override
  void dispose() {
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _initWhisper() async {
    try {
      setState(() => _error = null);

      // 1. Copy the bundled ggml-tiny.en model out of assets into a real
      //    file on disk (whisper.cpp loads models by file path).
      final Directory docsDir = await getApplicationDocumentsDirectory();
      final Directory modelDir = Directory('${docsDir.path}/whisper');
      if (!modelDir.existsSync()) {
        modelDir.createSync(recursive: true);
      }

      final File modelFile = File('${modelDir.path}/ggml-tiny.bin');
      if (!modelFile.existsSync() ||
          modelFile.lengthSync() != 77704715 /* tiny.en size */) {
        final ByteData assetBytes =
            await rootBundle.load('assets/models/ggml-tiny.en.bin');
        await modelFile.writeAsBytes(
          assetBytes.buffer.asUint8List(assetBytes.offsetInBytes),
          flush: true,
        );
      }

      // 2. Create the Whisper instance pointing at that directory.
      _whisper = Whisper(
        model: WhisperModel.tiny,
        modelDir: modelDir.path,
      );

      // 3. Sanity check the native library + warm up model loading.
      final String? version = await _whisper!.getVersion();
      debugPrint('whisper.cpp version: $version');

      if (mounted) {
        setState(() => _modelReady = true);
      }
    } catch (e) {
      if (mounted) {
        setState(() => _error = 'Model init failed: $e');
      }
    }
  }

  Future<void> _toggleRecording() async {
    if (_isTranscribing) return;

    if (_isRecording) {
      await _stopAndTranscribe();
      return;
    }

    try {
      setState(() {
        _error = null;
        _transcription = '';
      });

      // hasPermission() requests the RECORD_AUDIO runtime permission
      // if it has not been granted yet.
      if (!await _recorder.hasPermission()) {
        setState(() => _error = 'Microphone permission denied.');
        return;
      }

      final Directory docsDir = await getApplicationDocumentsDirectory();
      final String wavPath = '${docsDir.path}/recording.wav';

      // Remove any stale recording so we never transcribe old audio.
      final File oldFile = File(wavPath);
      if (oldFile.existsSync()) oldFile.deleteSync();

      await _recorder.start(_whisperRecordConfig, path: wavPath);
      setState(() => _isRecording = true);
    } catch (e) {
      setState(() => _error = 'Record start failed: $e');
    }
  }

  Future<void> _stopAndTranscribe() async {
    String? path;
    try {
      path = await _recorder.stop();
    } catch (e) {
      setState(() => _error = 'Record stop failed: $e');
    }

    setState(() {
      _isRecording = false;
      if (path == null) return;
      _isTranscribing = true;
    });
    if (path == null) return;

    try {
      final Stopwatch sw = Stopwatch()..start();
      final WhisperTranscribeResponse response = await _whisper!.transcribe(
        transcribeRequest: TranscribeRequest(
          audio: path,
          language: 'en', // tiny.en is English-only; 'auto' would fail.
          isTranslate: false,
          isNoTimestamps: true, // We only want the plain text.
          threads: 4,
          noFallback: false,
        ),
      );
      sw.stop();

      if (mounted) {
        setState(() {
          _transcription = response.text.trim();
          if (_transcription.isEmpty) {
            _transcription = '(No speech detected)';
          }
        });
      }
      debugPrint('Transcribed in ${sw.elapsedMilliseconds} ms');
    } catch (e) {
      if (mounted) {
        setState(() => _error = 'Transcription failed: $e');
      }
    } finally {
      if (mounted) setState(() => _isTranscribing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final ColorScheme cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Mumble Jumble'),
        backgroundColor: cs.surfaceContainerHighest,
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            spacing: 24,
            children: [
              // --- Overlay / accessibility setup card -----------------------
              _buildSetupCard(cs),

              // --- Status / transcription display ---------------------------
              Container(
                width: double.infinity,
                constraints: const BoxConstraints(minHeight: 160),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: cs.surfaceContainerHighest.withValues(alpha: 0.5),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: cs.outlineVariant),
                ),
                child: _buildStatusChild(cs),
              ),

              // --- Mic button ----------------------------------------------
              Center(
                child: GestureDetector(
                  onTap: (_modelReady && !_isTranscribing)
                      ? _toggleRecording
                      : null,
                  child: Container(
                    width: 96,
                    height: 96,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _isRecording
                          ? cs.errorContainer
                          : cs.primaryContainer,
                    ),
                    child: Icon(
                      _isRecording ? Icons.stop_rounded : Icons.mic_rounded,
                      size: 44,
                      color: _isRecording ? cs.onErrorContainer : cs.onPrimaryContainer,
                    ),
                  ),
                ),
              ),

              Text(
                _statusText,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: cs.onSurfaceVariant,
                    ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  String get _statusText {
    if (_isTranscribing) return 'Transcribing…';
    if (_isRecording) return 'Listening… tap to stop';
    if (!_modelReady) return 'Loading whisper model…';
    if (!_overlayPermissionReady) {
      return 'Enable the overlay permission to use the mic bubble';
    }
    return 'Tap the mic and start speaking';
  }

  /// Shows overlay-permission status and a deep link into the accessibility
  /// settings page where the user must enable our volume-key listener.
  Widget _buildSetupCard(ColorScheme cs) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: cs.outlineVariant),
      ),
      child: Row(
        children: [
          Icon(
            _overlayPermissionReady ? Icons.check_circle : Icons.info_outline,
            color: _overlayPermissionReady ? Colors.green : cs.primary,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _overlayPermissionReady
                  ? 'Overlay ready — double-press Volume Down anywhere '
                      'to pop the mic bubble\n(Enable "Mumble Jumble volume trigger" '
                      'in Accessibility settings for the global trigger)'
                  : 'Overlay permission needed for the floating mic bubble',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          TextButton(
            onPressed: _overlayPermissionReady
                ? () => _openAccessibilitySettings()
                : _requestOverlayPermission,
            child: Text(_overlayPermissionReady ? 'A11y' : 'Enable'),
          ),
        ],
      ),
    );
  }

  Future<void> _openAccessibilitySettings() async {
    const platform = MethodChannel('mumble_jumble/overlay_trigger');
    try {
      await platform.invokeMethod('openAccessibilitySettings');
    } on PlatformException catch (_) {
      // Fallback: generic accessibility settings.
      await platform.invokeMethod('openAccessibilitySettings');
    }
 }

  Widget _buildStatusChild(ColorScheme cs) {
    if (_error != null) {
      return Text(
        _error!,
        style: TextStyle(color: cs.error, fontWeight: FontWeight.w500),
      );
    }
    if (_isTranscribing) {
      return Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(
            width: 22,
            height: 22,
            child: CircularProgressIndicator(strokeWidth: 2.5),
          ),
          const SizedBox(width: 14),
          Text('Running whisper.cpp…', style: TextStyle(color: cs.onSurfaceVariant)),
        ],
      );
    }
    if (_transcription.isNotEmpty) {
      return Text(
        _transcription,
        style: Theme.of(context).textTheme.titleMedium,
      );
    }
    return Text(
      'Your transcription will appear here.',
      style: TextStyle(color: cs.onSurfaceVariant),
    );
  }
}
