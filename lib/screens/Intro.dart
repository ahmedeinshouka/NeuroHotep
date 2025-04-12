import 'package:flutter/material.dart';

class Intro extends StatefulWidget {
  const Intro({super.key});

  @override
  State<Intro> createState() => _IntroState();
}

class _IntroState extends State<Intro> {
  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        body: Stack(
          children: [
            // Background image
            Image.asset(
              'assets/images/background.png',
              fit: BoxFit.cover,
              width: double.infinity,
              height: double.infinity,
            ),
            // Logo positioned at the center-top
            Positioned(
              top: MediaQuery.of(context).size.height * 0.2,
              left: 0,
              right: 0,
              child: Center(
                child: Image.asset(
                  'assets/images/logo.png',
                  width: double.infinity,
                  height: 400,
                ),
              ),
            ),
            // Buttons positioned at the center-bottom
            Positioned(
              bottom: 100,
              left: 0,
              right: 0,
              child: Center(
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: () {
                        Navigator.pushNamed(context, '/scan'); // Navigate to scan screen
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF1A3C34),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(25),
                        ),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 20, vertical: 10), // Reduced padding
                        minimumSize: const Size(150, 60), // Fixed button size
                      ),
                      child: const Text(
                        'Scan',
                        style: TextStyle(
                          color: Colors.amber,
                          fontSize: 24, // Reduced font size for better fit
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    const SizedBox(width: 20),
                    ElevatedButton(
                      onPressed: () {
                        Navigator.pushNamed(context, '/upload'); // Navigate to upload screen
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF1A3C34),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(25),
                        ),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 20, vertical: 10), // Reduced padding
                        minimumSize: const Size(150, 60), // Fixed button size
                      ),
                      child: const Text(
                        'Upload',
                        style: TextStyle(
                          color: Colors.amber,
                          fontSize: 24, // Reduced font size for better fit
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}