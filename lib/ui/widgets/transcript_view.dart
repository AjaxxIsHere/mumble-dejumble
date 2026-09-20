import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../models/dictation_state.dart';

/// Presentation container for pipeline output: renders the error banner,
/// transcription progress, and the raw vs cleaned transcript sections based
/// solely on the immutable [DictationState].
class TranscriptView extends StatelessWidget {
  const TranscriptView({super.key, required this.state});

  final DictationState state;

  @override
  Widget build(BuildContext context) {
    final ColorScheme cs = Theme.of(context).colorScheme;
    TextEditingController _controller = TextEditingController();

    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(minHeight: 160),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: cs.outlineVariant),
      ),
      child: _buildChild(context, cs),
    );
  }

  Widget _buildChild(BuildContext context, ColorScheme cs) {
    if (state.errorMessage != null) {
      return Text(
        state.errorMessage!,
        style: TextStyle(color: cs.error, fontWeight: FontWeight.w500),
      );
    }

    if (state.status == DictationStatus.transcribing) {
      return _ProgressRow(cs: cs, label: 'Running whisper.cpp…');
    }

    if (state.rawTranscript.isNotEmpty) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Raw transcript', style: Theme.of(context).textTheme.labelLarge),
          const SizedBox(height: 6),
          Text(state.rawTranscript),
          if (state.status == DictationStatus.cleaning) ...[
            const SizedBox(height: 12),
            _ProgressRow(cs: cs, label: 'Polishing with the cleanup model…'),
          ],
          if (state.cleanedTranscript.isNotEmpty) ...[
            const SizedBox(height: 18),
            Text(
              'Cleaned transcript',
              style: Theme.of(context).textTheme.labelLarge,
            ),
            const SizedBox(height: 6),
            Text(
              state.cleanedTranscript,
              style: Theme.of(context).textTheme.titleMedium,
            ),
            Container(
              margin: const EdgeInsets.only(top: 8),
              child: IconButton(
                icon: const Icon(Icons.copy),
                onPressed: () {
                  Clipboard.setData(ClipboardData(text: state.cleanedTranscript));
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Copied to clipboard')),
                  );
                },
              ),
            ),
          ],
        ],
      );
    }

    return Text(
      'Your transcription will appear here.',
      style: TextStyle(color: cs.onSurfaceVariant),
    );
  }
}

class _ProgressRow extends StatelessWidget {
  const _ProgressRow({required this.cs, required this.label});

  final ColorScheme cs;
  final String label;

  @override
  Widget build(BuildContext context) {
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
        Text(label, style: TextStyle(color: cs.onSurfaceVariant)),
      ],
    );
  }
}
