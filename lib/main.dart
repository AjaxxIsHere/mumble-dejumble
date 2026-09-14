import 'package:flutter/material.dart';

import 'controllers/voice_controller.dart';
import 'screens/home_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mumble Jumble',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFFFA5D19),
          secondary: Color(0xFFFA5D19),
          surface: Color(0xFF171B20),
          onSurface: Color(0xFFE7E9EE),
        ),
        scaffoldBackgroundColor: const Color(0xFF121417),
      ),
      home: HomeScreen(controller: VoiceController()),
    );
  }
}