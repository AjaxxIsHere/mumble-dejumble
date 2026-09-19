import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_overlay_window/flutter_overlay_window.dart';

/// Channel the native [VolumeKeyAccessibilityService] uses to tell the main
/// isolate that the user double-pressed / single-pressed VOLUME_DOWN.
const MethodChannel _volumeTriggerChannel =
    MethodChannel('mumble_jumble/overlay_trigger');

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
/// Renders a compact circular mic button that floats at the top of the screen.
/// Tapping it (for now) just pulses - real voice capture will be wired in the
/// next milestone.
class MicBubble extends StatefulWidget {
  const MicBubble({super.key});

  @override
  State<MicBubble> createState() => _MicBubbleState();
}

class _MicBubbleState extends State<MicBubble> with SingleTickerProviderStateMixin {
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    return Material(
      type: MaterialType.transparency,
      child: Center(
        child: GestureDetector(
          onTapDown: (_) => setState(() => _pressed = true),
          onTapUp: (_) => setState(() => _pressed = false),
          onTapCancel: () => setState(() => _pressed = false),
          onTap: () async {
            // TODO: start/stop voice capture in a later milestone.
            await FlutterOverlayWindow.shareData('bubble_tap');
          },
          child: AnimatedScale(
            scale: _pressed ? 0.9 : 1.0,
            duration: const Duration(milliseconds: 120),
            child: Container(
              width: 72,
              height: 72,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.deepPurple.withValues(alpha: 0.95),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.35),
                    blurRadius: 14,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: const Icon(
                Icons.mic_rounded,
                color: Colors.white,
                size: 36,
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
  Future<bool> isPermissionGranted() => FlutterOverlayWindow.isPermissionGranted();

  /// Opens the system "Display over other apps" settings page for this app.
  /// Returns true once the user returns with the permission granted.
  Future<bool> requestPermission() async {
    final granted = await FlutterOverlayWindow.requestPermission();
    return granted ?? false;
  }

  /// Shows the mic bubble overlay at the top of the screen.
  Future<void> show() async {
    if (await FlutterOverlayWindow.isActive()) return;

    await FlutterOverlayWindow.showOverlay(
      height: 160,
      width: WindowSize.matchParent,
      alignment: OverlayAlignment.topCenter,
      flag: OverlayFlag.defaultFlag,
      enableDrag: true,
      positionGravity: PositionGravity.none,
      overlayTitle: 'Mumble Jumble listening',
      overlayContent: 'Tap the mic to start dictating. '
          'Volume-down once to dismiss.',
      visibility: NotificationVisibility.visibilityPublic,
    );
  }

  /// Hides the mic bubble overlay if it is active.
  Future<void> hide() async {
    if (!await FlutterOverlayWindow.isActive()) return;
    await FlutterOverlayWindow.closeOverlay();
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
