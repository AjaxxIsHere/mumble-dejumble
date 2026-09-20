import 'package:flutter/material.dart';

/// Circular recording trigger that animates between its idle (mic) and
/// recording (stop) appearances, with press-scale feedback.
///
/// Pure presentation: the parent decides when taps are allowed and what
/// they do.
class MicButton extends StatefulWidget {
  const MicButton({
    super.key,
    required this.recording,
    required this.enabled,
    this.onTap,
  });

  /// Whether audio is currently being captured (shows the stop glyph).
  final bool recording;

  /// Whether taps are accepted at all (false while busy / initializing).
  final bool enabled;

  final VoidCallback? onTap;

  @override
  State<MicButton> createState() => _MicButtonState();
}

class _MicButtonState extends State<MicButton> {
  bool _pressed = false;

  static const double _diameter = 96;
  static const double _iconSize = 44;

  @override
  Widget build(BuildContext context) {
    final ColorScheme cs = Theme.of(context).colorScheme;
    final Color background =
        widget.recording ? cs.errorContainer : cs.primaryContainer;
    final Color foreground =
        widget.recording ? cs.onErrorContainer : cs.onPrimaryContainer;
    final bool tappable = widget.enabled && widget.onTap != null;

    return AnimatedOpacity(
      opacity: tappable ? 1.0 : 0.55,
      duration: const Duration(milliseconds: 200),
      child: GestureDetector(
        onTap: tappable ? widget.onTap : null,
        onTapDown: tappable ? (_) => setState(() => _pressed = true) : null,
        onTapUp: (_) => setState(() => _pressed = false),
        onTapCancel: () => setState(() => _pressed = false),
        child: AnimatedScale(
          scale: _pressed ? 0.92 : 1.0,
          duration: const Duration(milliseconds: 120),
          curve: Curves.easeOut,
          child: AnimatedContainer(
            width: _diameter,
            height: _diameter,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOutCubic,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: background,
              boxShadow: [
                BoxShadow(
                  color: background.withValues(alpha: 0.4),
                  blurRadius: 18,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 200),
              child: Icon(
                widget.recording ? Icons.stop_rounded : Icons.mic_rounded,
                key: ValueKey<bool>(widget.recording),
                size: _iconSize,
                color: foreground,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
