package com.example.mumble_jumble

import android.content.Intent
import android.provider.Settings
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream

class MainActivity : FlutterActivity() {
    private val channelName = "mumble_jumble/assets"
    private val overlayTriggerChannelName = "mumble_jumble/overlay_trigger"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "ensureAssetCopied" -> {
                        val assetPath = call.argument<String>("assetPath")
                        val fileName = call.argument<String>("fileName")
                        if (assetPath == null || fileName == null) {
                            result.error("INVALID_ARGS", "assetPath and fileName are required", null)
                            return@setMethodCallHandler
                        }
                        try {
                            result.success(ensureAssetCopied(assetPath, fileName))
                        } catch (e: Exception) {
                            result.error("COPY_FAILED", e.message, null)
                        }
                    }
                    else -> result.notImplemented()
                }
            }

        // Overlay / accessibility helpers used by lib/overlay.dart.
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, overlayTriggerChannelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "openAccessibilitySettings" -> {
                        try {
                            val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                            startActivity(intent)
                            result.success(true)
                        } catch (e: Exception) {
                            result.error("SETTINGS_FAILED", e.message, null)
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    /**
     * Streams an asset to filesDir without buffering it fully in memory.
     * Writes to a .part file first and renames atomically, so an existing
     * destination file always means a complete copy.
     */
    private fun ensureAssetCopied(assetPath: String, fileName: String): String {
        val dest = File(filesDir, fileName)
        if (dest.exists() && dest.length() > 0) {
            return dest.absolutePath
        }

        val tmp = File(filesDir, "$fileName.part")
        tmp.delete()
        assets.open(assetPath).use { input ->
            FileOutputStream(tmp).use { output ->
                val buffer = ByteArray(1 shl 20) // 1 MiB chunks
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    output.write(buffer, 0, read)
                }
                output.flush()
            }
        }
        if (!tmp.renameTo(dest)) {
            tmp.delete()
            throw IllegalStateException("Could not finalize copied asset $fileName")
        }
        return dest.absolutePath
    }
}
