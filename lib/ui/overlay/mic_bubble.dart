import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_overlay_window/flutter_overlay_window.dart';

import '../../config/app_constants.dart';

/// The floating microphone bubble itself (Nothing-style yellow pill).
///
/// Renders a yellow capsule with a contrasting dark mic icon and an
/// undulating 4-bar waveform animation. Runs exclusively in the overlay
/// secondary isolate (see overlay_entry.dart).
class MicBubble extends StatefulWidget {
  const MicBubble({super.key});

  @override
  State<MicBubble> createState() => _MicBubbleState();
}

class _MicBubbleState extends State<MicBubble> with TickerProviderStateMixin {
  late final AnimationController _waveController;
  late final AnimationController _entryController;
  late final StreamSubscription<dynamic> _messageSubscription;
  bool _pressed = false;
  bool _isListening = true;
  bool _isDismissing = false;

  static const Duration _slideDuration = Duration(milliseconds: 260);
  static const double _pillHeight = 48;
  static const double _pillRadius = 24;

  @override
  void initState() {
    super.initState();
    _waveController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat();
    _entryController = AnimationController(
      vsync: this,
      duration: _slideDuration,
    )..forward();
    _messageSubscription = FlutterOverlayWindow.overlayListener.listen((data) {
      if (data == AppConstants.overlayHideMessage) {
        _dismiss();
      }
    });
  }

  @override
  void dispose() {
    _messageSubscription.cancel();
    _entryController.dispose();
    _waveController.dispose();
    super.dispose();
  }

  Future<void> _dismiss() async {
    if (_isDismissing) return;
    _isDismissing = true;
    _waveController.stop();
    await _entryController.reverse();
    await FlutterOverlayWindow.closeOverlay();
  }

  @override
  Widget build(BuildContext context) {
    return Material(
      type: MaterialType.transparency,
      child: Align(
        alignment: Alignment.centerRight,
        child: SlideTransition(
          position:
              Tween<Offset>(
                begin: const Offset(1.15, 0),
                end: Offset.zero,
              ).animate(
                CurvedAnimation(
                  parent: _entryController,
                  curve: Curves.easeOutCubic,
                ),
              ),
          child: GestureDetector(
            onTapDown: (_) => setState(() => _pressed = true),
            onTapUp: (_) => setState(() => _pressed = false),
            onTapCancel: () => setState(() => _pressed = false),
            onTap: () async {
              setState(() {
                _isListening = !_isListening;
                if (_isListening) {
                  _waveController.repeat();
                } else {
                  _waveController.stop();
                }
              });
              // Tell the main isolate the bubble was tapped.
              await FlutterOverlayWindow.shareData(
                AppConstants.overlayTapMessage,
              );
            },
            child: AnimatedScale(
              scale: _pressed ? 0.94 : 1.0,
              duration: const Duration(milliseconds: 120),
              curve: Curves.easeOut,
              child: Container(
                height: _pillHeight,
                padding: const EdgeInsets.symmetric(horizontal: 16),
                decoration: BoxDecoration(
                  color: AppConstants.overlayPillColor,
                  borderRadius: BorderRadius.circular(_pillRadius),
                  boxShadow: [
                    BoxShadow(
                      color: AppConstants.overlayPillColor
                          .withValues(alpha: 0.45),
                      blurRadius: 18,
                      offset: const Offset(0, 6),
                    ),
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.15),
                      blurRadius: 8,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.mic_rounded,
                      color: AppConstants.overlayAccentColor,
                      size: 22,
                    ),
                    const SizedBox(width: 10),
                    AnimatedBuilder(
                      animation: _waveController,
                      builder: (context, child) {
                        return Row(
                          mainAxisSize: MainAxisSize.min,
                          children: List.generate(4, (index) {
                            final phase = index * (math.pi / 2.5);
                            final wave = _isListening
                                ? (math.sin(
                                            _waveController.value *
                                                    2 *
                                                    math.pi +
                                                phase,
                                          ) +
                                          1) /
                                      2
                                : 0.15;
                            final barHeight = 6.0 + (wave * 18.0);

                            return Container(
                              margin:
                                  const EdgeInsets.symmetric(horizontal: 2),
                              width: 3.5,
                              height: barHeight,
                              decoration: BoxDecoration(
                                color: AppConstants.overlayAccentColor,
                                borderRadius: BorderRadius.circular(4),
                              ),
                            );
                          }),
                        );
                      },
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
