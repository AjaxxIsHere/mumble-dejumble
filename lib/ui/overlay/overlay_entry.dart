import 'package:flutter/material.dart';

import 'mic_bubble.dart';

/// OVERLAY SECONDARY ISOLATE — entry point boundary.
///
/// Everything reachable from [overlayMain] executes inside the secondary
/// Flutter engine spawned by flutter_overlay_window's `OverlayService`
/// (cached under `AppConstants.overlayEngineTag`). That engine has its own
/// memory space: it shares NOTHING with the main isolate except the platform
/// channels and the `overlayListener` message stream.
///
/// Rules for this subtree:
///  * No `DictationController`, no services, no main-isolate state.
///  * Communication with the main isolate happens only via
///    `FlutterOverlayWindow.shareData` / `overlayListener`.
///  * Main-isolate overlay lifecycle code lives in
///    `services/overlay_controller.dart`.
// library;

/// Entry point executed by the overlay's secondary Flutter engine.
///
/// flutter_overlay_window resolves this function *by name against the root
/// library* (DartEntrypoint(findAppBundlePath(), "overlayMain") in the
/// plugin's OverlayService.java), so `main.dart` re-exports this symbol.
/// The `@pragma('vm:entry-point')` is what keeps it alive in AOT builds.
@pragma('vm:entry-point')
void overlayMain() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const _OverlayApp());
}

/// Root widget of the overlay engine: a transparent, minimal Material app
/// hosting the floating mic bubble.
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
