import 'package:flutter/material.dart';

import 'controllers/dictation_controller.dart';
import 'services/audio_recorder_service.dart';
import 'services/overlay_controller.dart';
import 'services/speech_cleanup_service.dart';
import 'services/whisper_service.dart';
import 'ui/screens/voice_screen.dart';

/// Re-export the overlay engine's entry point from the root library.
///
/// flutter_overlay_window resolves `overlayMain` *by name* against the root
/// library (DartEntrypoint(findAppBundlePath(), "overlayMain") in the
/// plugin's OverlayService.java), but the function itself must stay
/// decoupled from app wiring and lives in `ui/overlay/overlay_entry.dart`
/// with its `@pragma('vm:entry-point')`. This export satisfies both: the
/// name is resolvable from the root library while the implementation remains
/// in the overlay entry file.
export 'ui/overlay/overlay_entry.dart' show overlayMain;

/// Composition root: constructs the headless services and the single
/// [DictationController] that binds them, then hands the controller to the
/// widget tree. No service or controller is ever constructed inside a
/// widget, which keeps the engines injectable into headless background
/// services later.
void main() {
  final controller = DictationController(
    audioRecorderService: AudioRecorderService(),
    whisperService: WhisperService(),
    speechCleanupService: SpeechCleanupService(),
    overlayController: OverlayController(),
  );

  runApp(MumbleJumbleApp(controller: controller));
}

class MumbleJumbleApp extends StatelessWidget {
  const MumbleJumbleApp({super.key, required this.controller});

  final DictationController controller;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mumble Jumble',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: VoiceScreen(controller: controller),
    );
  }
}
