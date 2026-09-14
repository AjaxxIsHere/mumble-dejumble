import 'package:permission_handler/permission_handler.dart';

class PermissionService {
  Future<bool> requestMicrophonePermission() async {
    final status = await Permission.microphone.request();
    return status.isGranted || status.isLimited;
  }

  Future<bool> get isMicrophoneGranted async {
    final status = await Permission.microphone.status;
    return status.isGranted || status.isLimited;
  }
}
