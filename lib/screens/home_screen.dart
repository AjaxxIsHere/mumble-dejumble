import 'package:flutter/material.dart';

import '../controllers/voice_controller.dart';
import '../models/app_state.dart';

class HomeScreen extends StatefulWidget {
  final VoiceController controller;

  const HomeScreen({super.key, required this.controller});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_handleControllerChange);
    widget.controller.initialize();
  }

  @override
  void dispose() {
    widget.controller.removeListener(_handleControllerChange);
    widget.controller.dispose();
    super.dispose();
  }

  void _handleControllerChange() {
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final state = controller.state;

    return Scaffold(
      backgroundColor: const Color(0xFF121417),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(height: 18),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: const [
                  Text(
                    'Local Voice AI',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.2,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              const Text(
                'Everything stays\non your device.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 18,
                  height: 1.3,
                  color: Colors.white70,
                ),
              ),
              const SizedBox(height: 28),
              Container(
                width: 110,
                height: 110,
                decoration: BoxDecoration(
                  color: state == VoiceAppState.recording
                      ? const Color(0xFFFF6A3D)
                      : const Color(0xFF1B1F24),
                  borderRadius: BorderRadius.circular(60),
                  border: Border.all(
                    color: state == VoiceAppState.recording
                        ? const Color(0xFFFA5D19)
                        : const Color(0xFF2C333A),
                    width: 2,
                  ),
                ),
                child: IconButton(
                  onPressed: (controller.isBusy && state != VoiceAppState.recording && state != VoiceAppState.transcribing && state != VoiceAppState.processing)
                      ? null
                      : () {
                          if (state == VoiceAppState.recording || state == VoiceAppState.transcribing || state == VoiceAppState.processing) {
                            controller.stopListening();
                          } else {
                            controller.startListening();
                          }
                        },
                  icon: Icon(
                    state == VoiceAppState.recording || state == VoiceAppState.transcribing || state == VoiceAppState.processing
                        ? Icons.stop_rounded
                        : Icons.mic_none_rounded,
                    size: 40,
                    color: state == VoiceAppState.recording || state == VoiceAppState.transcribing || state == VoiceAppState.processing
                        ? Colors.white
                        : const Color(0xFFFA5D19),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Text(
                _statusLabel(state),
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFFFA5D19),
                ),
              ),
              const SizedBox(height: 28),
              if (controller.errorMessage.isNotEmpty)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: const Color(0xFF251914),
                    border: Border.all(color: const Color(0xFFFA5D19)),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(controller.errorMessage),
                ),
              if (controller.showRawTranscript && controller.rawTranscript.isNotEmpty)
                _TranscriptCard(label: 'Raw transcript', text: controller.rawTranscript),
              const SizedBox(height: 16),
              _TranscriptCard(label: 'Polished text', text: controller.polishedText.isEmpty ? 'Ready when you are.' : controller.polishedText),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton(
                      onPressed: controller.polishedText.isEmpty ? null : () => controller.copyResult(),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFFFA5D19),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                      ),
                      child: const Text('Copy'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () {},
                      style: OutlinedButton.styleFrom(
                        side: const BorderSide(color: Color(0xFFFA5D19)),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                      ),
                      child: const Text('Edit'),
                    ),
                  ),
                ],
              ),
              const Spacer(),
              const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.fiber_manual_record, size: 12, color: Color(0xFFFA5D19)),
                  SizedBox(width: 6),
                  Text('On-device', style: TextStyle(color: Colors.white70)),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  String _statusLabel(VoiceAppState state) {
    switch (state) {
      case VoiceAppState.initializingStt:
        return 'Loading local STT…';
      case VoiceAppState.initializingLlm:
        return 'Loading local model…';
      case VoiceAppState.ready:
        return 'Tap to speak';
      case VoiceAppState.recording:
        return 'Listening…';
      case VoiceAppState.transcribing:
        return 'Transcribing…';
      case VoiceAppState.processing:
        return 'Processing…';
      case VoiceAppState.completed:
        return 'Complete';
      case VoiceAppState.error:
        return 'Error';
      case VoiceAppState.idle:
        return 'Idle';
    }
  }
}

class _TranscriptCard extends StatelessWidget {
  final String label;
  final String text;

  const _TranscriptCard({required this.label, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF171B20),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFF2C333A)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 12,
              color: Colors.white54,
              letterSpacing: 0.6,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            text,
            style: const TextStyle(
              fontSize: 18,
              height: 1.5,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}
