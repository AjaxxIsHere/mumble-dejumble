import 'package:flutter/material.dart';

/// Setup card that reports overlay-permission status and offers the matching
/// action: request the permission first, then deep-link into the
/// accessibility settings where the user enables the volume-key trigger.
///
/// Pure presentation: callbacks are provided by the owning screen.
class SetupCard extends StatelessWidget {
  const SetupCard({
    super.key,
    required this.overlayPermissionGranted,
    required this.onRequestOverlayPermission,
    required this.onOpenAccessibilitySettings,
  });

  final bool overlayPermissionGranted;
  final VoidCallback onRequestOverlayPermission;
  final VoidCallback onOpenAccessibilitySettings;

  @override
  Widget build(BuildContext context) {
    final ColorScheme cs = Theme.of(context).colorScheme;
    final TextTheme textTheme = Theme.of(context).textTheme;

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: cs.outlineVariant),
      ),
      child: Row(
        children: [
          Icon(
            overlayPermissionGranted ? Icons.check_circle : Icons.info_outline,
            color: overlayPermissionGranted ? Colors.green : cs.primary,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              overlayPermissionGranted
                  ? 'Overlay ready — double-press Volume Down anywhere '
                        'to pop the mic bubble\n(Enable "Mumble Jumble volume trigger" '
                        'in Accessibility settings for the global trigger)'
                  : 'Overlay permission needed for the floating mic bubble',
              style: textTheme.bodySmall,
            ),
          ),
          TextButton(
            onPressed: overlayPermissionGranted
                ? onOpenAccessibilitySettings
                : onRequestOverlayPermission,
            child: Text(overlayPermissionGranted ? 'A11y' : 'Enable'),
          ),
        ],
      ),
    );
  }
}
