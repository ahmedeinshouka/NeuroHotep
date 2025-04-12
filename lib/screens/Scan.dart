import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'dart:io';

class scan extends StatelessWidget {
  final List<CameraDescription> cameras;
  const scan({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    // Remove the MaterialApp and directly return TranslationScreen
    return TranslationScreen(cameras: cameras);
  }
}

class TranslationScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const TranslationScreen({super.key, required this.cameras});

  @override
  _TranslationScreenState createState() => _TranslationScreenState();
}

class _TranslationScreenState extends State<TranslationScreen> {
  String _scannedText = 'This is the translated text that might be very long and needs to be scrollable. You can add more text here to test the scrolling functionality.';
  String _translatedText = 'This is another block of text that might be very long and needs to be scrollable. Add more text here to see how it scrolls.';

  // Camera-related variables
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool _isCameraPreviewVisible = false; // Toggle to show/hide camera preview

  @override
  void initState() {
    super.initState();
    // Initialize the camera controller
    _controller = CameraController(
      widget.cameras.first, // Use the first available camera
      ResolutionPreset.high,
    );
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  // Function to toggle the camera preview
  void _toggleCameraPreview() {
    setState(() {
      _isCameraPreviewVisible = !_isCameraPreviewVisible;
    });
  }

  // Function to capture an image and perform OCR
  Future<void> _captureImage() async {
    try {
      await _initializeControllerFuture;
      final image = await _controller.takePicture();
      final extractedText = await _performOCR(image.path);
      setState(() {
        _scannedText = extractedText; // Update the scanned text
        _isCameraPreviewVisible = false; // Hide the camera preview after capturing
      });
    } catch (e) {
      print("Error capturing image: $e");
      setState(() {
        _scannedText = 'Error capturing image.';
        _isCameraPreviewVisible = false;
      });
    }
  }

  // Function to perform OCR using Google ML Kit
  Future<String> _performOCR(String imagePath) async {
    final inputImage = InputImage.fromFilePath(imagePath);
    final textRecognizer = GoogleMlKit.vision.textRecognizer();
    final recognizedText = await textRecognizer.processImage(inputImage);

    String extractedText = recognizedText.text;

    await textRecognizer.close();
    return extractedText.isEmpty ? 'No text detected.' : extractedText;
  }

  @override
  Widget build(BuildContext context) {
    const double containerHeight = 350;
    const double containerWidth = 370; // Width based on aspect ratio

    // Define fixed dimensions for the text containers
    const double textContainerHeight = 100;
    const double textContainerWidth = double.infinity; // Full width of the parent

    return SafeArea(
      child: Scaffold(
        body: Stack(
          children: [
            // Background with corrected color
            Container(
              color: const Color.fromARGB(255, 15, 37, 32),
              width: double.infinity,
              height: double.infinity,
            ),
            // Camera preview or static image
            Positioned(
              top: 20,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  width: containerWidth, // Fixed width based on aspect ratio
                  height: containerHeight, // Fixed height
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.amber, width: 3),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(20),
                    child: _isCameraPreviewVisible
                        ? FutureBuilder<void>(
                            future: _initializeControllerFuture,
                            builder: (context, snapshot) {
                              if (snapshot.connectionState == ConnectionState.done) {
                                return AspectRatio(
                                  aspectRatio: 4 / 3,
                                  child: CameraPreview(_controller),
                                );
                              } else {
                                return const Center(child: CircularProgressIndicator());
                              }
                            },
                          )
                        : Image.asset(
                            'assets/images/translate.png',
                            width: containerWidth,
                            height: containerHeight,
                            fit: BoxFit.cover, // Ensure the image scales properly
                          ),
                  ),
                ),
              ),
            ),
            // Main content (text containers and buttons)
            Positioned.fill(
              top: 400,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // First text container (scrollable) - Before Translation
                    SizedBox(
                      width: textContainerWidth, // Fixed width
                      height: textContainerHeight, // Fixed height
                      child: Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: Colors.amber, width: 3),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.2),
                              blurRadius: 5,
                              offset: const Offset(0, 3),
                            ),
                          ],
                        ),
                        child: SingleChildScrollView(
                          padding: const EdgeInsets.all(15),
                          child: Text(
                            _scannedText, // Display the scanned text
                            style: const TextStyle(
                              color: Colors.black,
                              fontSize: 16,
                            ),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                    // Language label
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: Colors.amber, width: 3),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.2),
                            blurRadius: 5,
                            offset: const Offset(0, 3),
                          ),
                        ],
                      ),
                      child: const Text(
                        'English',
                        style: TextStyle(
                          color: Colors.black,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                    // Second text container (scrollable) - After Translation
                    SizedBox(
                      width: textContainerWidth, // Fixed width
                      height: textContainerHeight, // Fixed height
                      child: Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: Colors.amber, width: 3),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.2),
                              blurRadius: 5,
                              offset: const Offset(0, 3),
                            ),
                          ],
                        ),
                        child: SingleChildScrollView(
                          padding: const EdgeInsets.all(15),
                          child: Text(
                            _translatedText, // Display translated text
                            style: const TextStyle(
                              color: Colors.black,
                              fontSize: 16,
                            ),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                    // Buttons
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          decoration: BoxDecoration(
                            gradient: const LinearGradient(
                              colors: [Colors.white, Colors.white],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderRadius: BorderRadius.circular(30),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.2),
                                blurRadius: 5,
                                offset: const Offset(0, 3),
                              ),
                            ],
                          ),
                          child: ElevatedButton(
                            onPressed: () {
                              if (_isCameraPreviewVisible) {
                                _captureImage(); // Capture the image if camera is visible
                              } else {
                                _toggleCameraPreview(); // Show the camera preview
                              }
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.transparent,
                              shadowColor: Colors.transparent,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(30),
                              ),
                              padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                              minimumSize: const Size(150, 60),
                            ),
                            child: Text(
                              _isCameraPreviewVisible ? 'CAPTURE' : 'SCAN',
                              style: const TextStyle(
                                color: Colors.amber,
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 20),
                        Container(
                          decoration: BoxDecoration(
                            gradient: const LinearGradient(
                              colors: [Colors.white, Colors.white],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderRadius: BorderRadius.circular(30),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.2),
                                blurRadius: 5,
                                offset: const Offset(0, 3),
                              ),
                            ],
                          ),
                          child: ElevatedButton(
                            onPressed: () {
                              print("Translate button pressed");
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.transparent,
                              shadowColor: Colors.transparent,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(30),
                              ),
                              padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                              minimumSize: const Size(150, 60),
                            ),
                            child: const Text(
                              'TRANSLATE',
                              style: TextStyle(
                                color: Colors.amber,
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                      ],
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