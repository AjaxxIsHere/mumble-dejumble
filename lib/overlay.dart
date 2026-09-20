import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_overlay_window/flutter_overlay_window.dart';

/// Channel the native [VolumeKeyAccessibilityService] uses to tell the main
/// isolate that the user double-pressed / single-pressed VOLUME_DOWN.
const MethodChannel _volumeTriggerChannel = MethodChannel(
  'mumble_jumble/overlay_trigger',
);

/// The cached overlay engine tag used by flutter_overlay_window internally.
/// We mirror it here so the accessibility service and Dart agree on it.
const String kOverlayEngineTag = 'myCachedEngine';

// ---------------------------------------------------------------------------
//  Overlay entry point (runs in the overlay isolate)
// ---------------------------------------------------------------------------

/// Starts the overlay widget tree from the root-library entry point in
/// main.dart, as required by flutter_overlay_window.
void runOverlayApp() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const _OverlayApp());
}

class _OverlayApp extends StatelessWidget {
  const _OverlayApp();

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true),
      home: const MicBubble(),
    );
  }
}

/// The floating microphone bubble itself.
///
/// Renders a yellow (#F7C000) pill capsule with a contrasting dark mic icon
/// and an undulating 4-bar waveform animation.
class MicBubble extends StatefulWidget {
  const MicBubble({super.key});

  @override
  State<MicBubble> createState() => _MicBubbleState();
}

class _MicBubbleState extends State<MicBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _waveController;
  late final AnimationController _entryController;
  late final StreamSubscription<dynamic> _messageSubscription;
  bool _pressed = false;
  bool _isListening = true;
  bool _isDismissing = false;

  static const Color _pillColor = Color(0xFFF7C000);
  static const Color _accentColor = Color(0xFF18181B);
  static const Duration _slideDuration = Duration(milliseconds: 260);

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
      if (data == 'hide_overlay') {
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
              await FlutterOverlayWindow.shareData('bubble_tap');
            },
            child: AnimatedScale(
              scale: _pressed ? 0.94 : 1.0,
              duration: const Duration(milliseconds: 120),
              curve: Curves.easeOut,
              child: Container(
                height: 48,
                padding: const EdgeInsets.symmetric(horizontal: 16),
                decoration: BoxDecoration(
                  color: _pillColor,
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: _pillColor.withValues(alpha: 0.45),
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
                      color: _accentColor,
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
                              margin: const EdgeInsets.symmetric(horizontal: 2),
                              width: 3.5,
                              height: barHeight,
                              decoration: BoxDecoration(
                                color: _accentColor,
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

// ---------------------------------------------------------------------------
//  Overlay controller (runs in the main isolate)
// ---------------------------------------------------------------------------

/// Controls the overlay bubble lifecycle from the main isolate.
///
/// Usage in main.dart:
/// ```dart
/// final controller = OverlayController();
/// await controller.initialize();   // requests permissions if needed
/// await controller.show();         // e.g. on volume double-press
/// await controller.hide();         // e.g. on volume single-press
/// ```
class OverlayController {
  StreamSubscription<dynamic>? _overlayListenerSub;

  /// Whether the overlay permission (SYSTEM_ALERT_WINDOW) is granted.
  Future<bool> isPermissionGranted() =>
      FlutterOverlayWindow.isPermissionGranted();

  /// Opens the system "Display over other apps" settings page for this app.
  /// Returns true once the user returns with the permission granted.
  Future<bool> requestPermission() async {
    final granted = await FlutterOverlayWindow.requestPermission();
    return granted ?? false;
  }

  /// Shows the mic bubble beside the phone's right-side volume controls.
  Future<void> show() async {
    if (await FlutterOverlayWindow.isActive()) return;

    await FlutterOverlayWindow.showOverlay(
      height: 96,
      width: 220,
      alignment: OverlayAlignment.centerRight,
      flag: OverlayFlag.defaultFlag,
      enableDrag: true,
      positionGravity: PositionGravity.none,
      overlayTitle: 'Mumble Jumble listening',
      overlayContent:
          'Tap the mic to start dictating. '
          'Volume-down once to dismiss.',
      visibility: NotificationVisibility.visibilityPublic,
    );
  }

  /// Hides the mic bubble overlay if it is active.
  Future<void> hide() async {
    if (!await FlutterOverlayWindow.isActive()) return;
    await FlutterOverlayWindow.shareData('hide_overlay');
    await Future<void>.delayed(const Duration(milliseconds: 300));
    if (await FlutterOverlayWindow.isActive()) {
      await FlutterOverlayWindow.closeOverlay();
    }
  }

  /// Listens for messages broadcast from the overlay isolate
  /// (e.g. the 'bubble_tap' emitted by [MicBubble]).
  Stream<dynamic> get messages => FlutterOverlayWindow.overlayListener;

  /// Call from the main isolate once at startup.
  ///
  /// Wires up:
  ///  1. The native volume-key accessibility service channel
  ///     (double press -> show, single press -> hide).
  ///  2. Sync of overlay active state back to the native side so the
  ///     key handler knows whether a single press should hide the bubble.
  Future<void> initialize() async {
    _volumeTriggerChannel.setMethodCallHandler((call) async {
      switch (call.method) {
        case 'volumeTrigger':
          final action = call.arguments as String?;
          if (action == 'show') {
            await show();
          } else if (action == 'hide') {
            await hide();
          }
        case 'isOverlayActive':
          final active = await FlutterOverlayWindow.isActive();
          return active;
      }
      return null;
    });

    // Keep the native side informed about bubble visibility so
    // [VolumeKeyAccessibilityService.isOverlayActive] stays in sync.
    _overlayListenerSub?.cancel();
    _overlayListenerSub = null;
  }

  /// Clean up listeners. Does not close the overlay itself.
  void dispose() {
    _volumeTriggerChannel.setMethodCallHandler(null);
    _overlayListenerSub?.cancel();
    _overlayListenerSub = null;
  }
}
