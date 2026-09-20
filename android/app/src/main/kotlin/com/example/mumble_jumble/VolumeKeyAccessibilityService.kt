package com.example.mumble_jumble

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.KeyEvent
import android.view.accessibility.AccessibilityEvent
import flutter.overlay.window.flutter_overlay_window.OverlayService
import io.flutter.embedding.engine.FlutterEngineCache
import io.flutter.plugin.common.MethodChannel

/**
 * System-wide volume-key listener.
 *
 * Android routes volume-key events to the focused window before any app can
 * observe them, so the only reliable way to catch KEYCODE_VOLUME_DOWN over
 * other apps is an AccessibilityService (same technique used by
 * "button remapper" apps). With FLAG_REQUEST_FILTER_KEY_EVENTS the service
 * receives every key event through onKeyEvent() and can choose to consume it.
 *
 * Behaviour:
 *  - Double press VOLUME_DOWN (two presses within 600 ms)  -> show overlay bubble
 *  - Single VOLUME_DOWN press while the bubble is active    -> hide overlay bubble
 *
 * The service runs in the same process as the Flutter engine, so it talks to
 * Dart over a MethodChannel on the main engine
 * ("mumble_jumble/overlay_trigger"), and reads the overlay state directly
 * from [OverlayService.isRunning].
 */
class VolumeKeyAccessibilityService : AccessibilityService() {

    companion object {
        private const val TAG = "VolumeKeyA11y"

        /// Must match the engine tag used by flutter_overlay_window
        /// (OverlayConstants.CACHED_TAG).
        private const val OVERLAY_ENGINE_TAG = "myCachedEngine"

        /// Must match lib/overlay.dart.
        private const val TRIGGER_CHANNEL = "mumble_jumble/overlay_trigger"

        /// Two volume-down presses inside this window count as a double press.
        private const val DOUBLE_PRESS_WINDOW_MS = 600L

        /// Handler delay used to disambiguate a single press from the first
        /// press of a double press before hiding the bubble.
        private const val SINGLE_PRESS_DELAY_MS = DOUBLE_PRESS_WINDOW_MS + 50L

        /// Must outlast the overlay bubble's slide-out animation.
        private const val OVERLAY_EXIT_DELAY_MS = 300L

        @Volatile
        var instance: VolumeKeyAccessibilityService? = null
    }

    private var methodChannel: MethodChannel? = null
    private var lastVolumeDownAt: Long = 0L
    private val handler = Handler(Looper.getMainLooper())

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.i(TAG, "Volume key accessibility service connected")
    }

    override fun onDestroy() {
        instance = null
        super.onDestroy()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // Key events arrive via onKeyEvent(); nothing to do here.
    }

    override fun onInterrupt() {
        // Required override; no special behaviour.
    }

    override fun onKeyEvent(event: KeyEvent): Boolean {
        if (event.action != KeyEvent.ACTION_DOWN ||
            event.keyCode != KeyEvent.KEYCODE_VOLUME_DOWN
        ) {
            return false
        }

        // Ignore auto-repeat from a held press.
        if (event.repeatCount > 0) return true

        val now = SystemClock.elapsedRealtime()
        val isDoublePress = now - lastVolumeDownAt <= DOUBLE_PRESS_WINDOW_MS
        lastVolumeDownAt = now

        if (isDoublePress) {
            // Second press of a double press: cancel any pending "hide"
            // decision and show the bubble.
            handler.removeCallbacksAndMessages(null)
            if (!OverlayService.isRunning) {
                Log.i(TAG, "Volume-down double press -> show overlay bubble")
                notifyDart("show")
                startOverlayService()
            } else {
                Log.d(TAG, "Double press while bubble already visible; ignoring")
            }
        } else if (OverlayService.isRunning) {
            // First press while the bubble is visible. Wait one beat so a
            // genuine double press is never mistaken for a hide command;
            // if no second press arrives, hide the bubble.
            handler.postDelayed({
                val stillSingle =
                    SystemClock.elapsedRealtime() - lastVolumeDownAt >= DOUBLE_PRESS_WINDOW_MS
                if (stillSingle && OverlayService.isRunning) {
                    Log.i(TAG, "Volume-down single press -> hide overlay bubble")
                    notifyDart("hide")
                    stopOverlayService()
                }
            }, SINGLE_PRESS_DELAY_MS)
        }

        // Consume the event so the system volume UI does not pop up.
        return true
    }

    // -------------------------------------------------------------------------
    //  Bridge to Flutter
    // -------------------------------------------------------------------------

    /// Notify the main isolate so Dart-side state (OverlayController) can
    /// react. The cached overlay engine tag is used because the main engine
    /// and overlay engine share the same process; the main engine runs on
    /// the default DartEntrypoint ("main").
    private fun notifyDart(action: String) {
        val engine = FlutterEngineCache.getInstance().get(OVERLAY_ENGINE_TAG)
        if (engine != null) {
            try {
                MethodChannel(
                    engine.dartExecutor.binaryMessenger,
                    TRIGGER_CHANNEL
                ).invokeMethod("volumeTrigger", action)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to notify Dart: $e")
            }
        }
    }

    // -------------------------------------------------------------------------
    //  Native overlay service control (fallback so the trigger works even if
    //  the main isolate is busy)
    // -------------------------------------------------------------------------

    private fun startOverlayService() {
        try {
            val intent = Intent(this, OverlayService::class.java)
            intent.putExtra("startX", 0)
            intent.putExtra("startY", 0)
            startService(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start OverlayService from accessibility service: $e")
        }
    }

    private fun stopOverlayService() {
        try {
            val intent = Intent(this, OverlayService::class.java)
            intent.putExtra(OverlayService.INTENT_EXTRA_IS_CLOSE_WINDOW, true)
            handler.postDelayed({
                try {
                    startService(intent)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to stop OverlayService after animation: $e")
                }
            }, OVERLAY_EXIT_DELAY_MS)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop OverlayService from accessibility service: $e")
        }
    }
}
