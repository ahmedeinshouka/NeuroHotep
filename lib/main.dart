import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'screens/Splash.dart';
import 'screens/Intro.dart';
import 'screens/Scan.dart';
import 'screens/upload.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  // Initialize the camera for the Scan screen
  final cameras = await availableCameras();

  runApp(Neurohotep(initialRoute: '/', cameras: cameras));
}

class Neurohotep extends StatelessWidget {
  final String initialRoute;
  final List<CameraDescription> cameras;

  const Neurohotep({super.key, required this.initialRoute, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false, // This hides the debug banner
      initialRoute: initialRoute,
      routes: {
        '/': (context) => const SplashScreen(),
        '/home': (context) => const Intro(),
        '/scan': (context) => scan(cameras: cameras),
        '/upload': (context) => const UploadScreen(),
      },
      onUnknownRoute: (settings) {
        return MaterialPageRoute(
          builder: (context) => Scaffold(
            body: Center(
              child: Text('Route "${settings.name}" not found.'),
            ),
          ),
        );
      },
    );
  }
}