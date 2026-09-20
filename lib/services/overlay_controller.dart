import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_overlay_window/flutter_overlay_window.dart';

import '../config/app_constants.dart';

/// MAIN-ISOLATE lifecycle coordinator for the overlay bubble.
///
/// This class owns every overlay concern that runs in the main Flutter
/// isolate: the SYSTEM_ALERT_WINDOW permission flow, showing/hiding the
/// bubble, and the method channel that the native volume-key accessibility
/// service uses to trigger it.
///
/// It must never be imported from overlay-isolate code — the secondary
/// engine entry point lives in `ui/overlay/overlay_entry.dart` and has its
/// own memory space (see the isolate-boundary notes there).
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
      height: AppConstants.overlayHeight,
      width: AppConstants.overlayWidth,
      alignment: OverlayAlignment.centerRight,
      flag: OverlayFlag.defaultFlag,
      enableDrag: true,
      positionGravity: PositionGravity.none,
      overlayTitle: 'Mumble Jumble listening',
      overlayContent:
          'Tap the mic to start dictating. Volume-down once to dismiss.',
      visibility: NotificationVisibility.visibilityPublic,
    );
  }

  /// Hides the mic bubble overlay if it is active.
  ///
  /// Asks the bubble to play its exit animation first, then closes the
  /// overlay engine after [AppConstants.overlayExitDelay].
  Future<void> hide() async {
    if (!await FlutterOverlayWindow.isActive()) return;
    await FlutterOverlayWindow.shareData(AppConstants.overlayHideMessage);
    await Future<void>.delayed(AppConstants.overlayExitDelay);
    if (await FlutterOverlayWindow.isActive()) {
      await FlutterOverlayWindow.closeOverlay();
    }
  }

  /// Listens for messages broadcast from the overlay isolate
  /// (e.g. the 'bubble_tap' emitted by the [MicBubble] widget).
  Stream<dynamic> get messages => FlutterOverlayWindow.overlayListener;

  /// Call from the main isolate once at startup.
  ///
  /// Wires the native volume-key accessibility service channel:
  ///  - `volumeTrigger` with 'show'/'hide' arguments,
  ///  - `isOverlayActive` so the key handler can query bubble visibility.
  Future<void> initialize() async {
    const MethodChannel(AppConstants.overlayTriggerChannel)
        .setMethodCallHandler((call) async {
      switch (call.method) {
        case AppConstants.volumeTriggerMethod:
          final action = call.arguments as String?;
          if (action == AppConstants.volumeTriggerShow) {
            await show();
          } else if (action == AppConstants.volumeTriggerHide) {
            await hide();
          }
        case AppConstants.isOverlayActiveMethod:
          return FlutterOverlayWindow.isActive();
      }
      return null;
    });

    _overlayListenerSub?.cancel();
    _overlayListenerSub = null;
  }

  /// Deep-links into the system accessibility settings page, where the user
  /// must enable the "Mumble Jumble volume trigger" service for the global
  /// volume-key gesture to work.
  Future<void> openAccessibilitySettings() async {
    const platform = MethodChannel(AppConstants.overlayTriggerChannel);
    try {
      await platform
          .invokeMethod<void>(AppConstants.openAccessibilitySettingsMethod);
    } on PlatformException catch (e) {
      debugPrint('Could not open accessibility settings: $e');
    }
  }

  /// Clean up listeners. Does not close the overlay itself.
  void dispose() {
    const MethodChannel(AppConstants.overlayTriggerChannel)
        .setMethodCallHandler(null);
    _overlayListenerSub?.cancel();
    _overlayListenerSub = null;
  }
}
