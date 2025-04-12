import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'dart:async';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  bool _showButton = false;
  double _progress = 0.0;
  late Timer _timer;
  bool _isPlaying = true; // Music state
  final AudioPlayer _audioPlayer = AudioPlayer()..setReleaseMode(ReleaseMode.loop);

  @override
  void initState() {
    super.initState();

    // Start playing background music
    _playMusic();

    // Start a timer that updates progress smoothly over 10 seconds
    _timer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      setState(() {
        if (_progress < 1) {
          _progress += 0.01; // Increase progress gradually
        } else {
          _showButton = true;
          _timer.cancel(); // Stop the timer when progress is complete
        }
      });
    });
  }

  // Function to play music
  Future<void> _playMusic() async {
    await _audioPlayer.setSource(AssetSource('audio/background.mp3')); // Play from assets
    await _audioPlayer.resume();
  }

  // Function to toggle music on/off
  void _toggleMusic() {
    if (_isPlaying) {
      _audioPlayer.pause();
    } else {
      _audioPlayer.resume();
    }
    setState(() {
      _isPlaying = !_isPlaying;
    });
  }

  @override
  void dispose() {
    _timer.cancel(); // Stop timer
    _audioPlayer.stop(); // Stop music when screen is disposed
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        body: Stack(
          children: [
            Image.asset(
              'assets/images/intro.jpg',
              fit: BoxFit.fill,
              width: double.infinity,
              height: double.infinity,
            ),
            Positioned(
              top: 18,
              right: 7,
              child: IconButton(
                icon: Icon(
                  _isPlaying ? Icons.volume_up : Icons.volume_off,
                  color: Color.fromRGBO(172, 130, 22, 1),
                  size: 30,
                ),
                onPressed: _toggleMusic,
              ),
            ),
            Positioned(
              bottom: 100,
              left: 0,
              right: 0,
              child: Column(
                children: [
                  if (!_showButton)
                    Column(
                      children: [
                        CircularProgressIndicator(
                          value: _progress,
                          color: Colors.white,
                          strokeWidth: 5,
                        ),
                        const SizedBox(height: 10),
                        Text(
                          "${(_progress * 100).toInt()}%",
                          style: const TextStyle(color: Colors.white),
                        ),
                      ],
                    ),
                  const SizedBox(height: 10),
                  if (_showButton)
                    ElevatedButton(
                      onPressed: () {
                        Navigator.pushNamed(context, '/home'); // Navigate to home screen
                      },
                      child: const Text(
                        "Start Your Journey",
                        style: TextStyle(
                          color: Color.fromRGBO(172, 130, 22, 1),
                          fontWeight: FontWeight.bold,
                          fontSize: 20,
                        ),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(
                            horizontal: 40, vertical: 15),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(30),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
