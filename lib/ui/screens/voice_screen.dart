import 'package:flutter/material.dart';

import '../../controllers/dictation_controller.dart';
import '../../models/dictation_state.dart';
import '../widgets/mic_button.dart';
import '../widgets/setup_card.dart';
import '../widgets/transcript_view.dart';

/// Main app screen. Listens to the [DictationController] through a
/// [ListenableBuilder] and delegates all behaviour to it; the screen itself
/// owns no application state beyond the controller reference it is given.
class VoiceScreen extends StatefulWidget {
  const VoiceScreen({super.key, required this.controller});

  /// App-scoped controller wired in `main.dart`.
  final DictationController controller;

  @override
  State<VoiceScreen> createState() => _VoiceScreenState();
}

class _VoiceScreenState extends State<VoiceScreen> {
  @override
  void initState() {
    super.initState();
    // Idempotent; UI reacts to progress via ListenableBuilder.
    widget.controller.initialize();
  }

  @override
  Widget build(BuildContext context) {
    final ColorScheme cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Mumble Jumble'),
        backgroundColor: cs.surfaceContainerHighest,
      ),
      body: ListenableBuilder(
        listenable: widget.controller,
        builder: (context, _) {
          final DictationState state = widget.controller.state;
          return Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                spacing: 24,
                children: [
                  // --- Overlay / accessibility setup card -----------------
                  SetupCard(
                    overlayPermissionGranted: state.overlayPermissionGranted,
                    onRequestOverlayPermission:
                        widget.controller.requestOverlayPermission,
                    onOpenAccessibilitySettings:
                        widget.controller.openAccessibilitySettings,
                  ),

                  // --- Status / transcription display ---------------------
                  TranscriptView(state: state),

                  // --- Mic button -----------------------------------------
                  Center(
                    child: MicButton(
                      recording:
                          state.status == DictationStatus.recording,
                      enabled: state.canToggleRecording,
                      onTap: widget.controller.toggleRecording,
                    ),
                  ),

                  Text(
                    _statusText(state),
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: cs.onSurfaceVariant,
                        ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  String _statusText(DictationState state) {
    switch (state.status) {
      case DictationStatus.initializing:
        return 'Loading whisper model…';
      case DictationStatus.failed:
        return 'Model init failed — restart the app to retry';
      case DictationStatus.recording:
        return 'Listening… tap to stop';
      case DictationStatus.transcribing:
        return 'Transcribing…';
      case DictationStatus.cleaning:
        return 'Cleaning up the transcript…';
      case DictationStatus.idle:
        return state.overlayPermissionGranted
            ? 'Tap the mic and start speaking'
            : 'Enable the overlay permission to use the mic bubble';
    }
  }
}
