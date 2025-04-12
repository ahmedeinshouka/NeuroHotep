import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

class UploadScreen extends StatelessWidget {
  const UploadScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // Remove the MaterialApp and directly return TranslationScreen
    return const TranslationScreen();
  }
}

class TranslationScreen extends StatefulWidget {
  const TranslationScreen({super.key});

  @override
  _TranslationScreenState createState() => _TranslationScreenState();
}

class _TranslationScreenState extends State<TranslationScreen> {
  String _scannedText = 'This is the translated text that might be very long and needs to be scrollable. You can add more text here to test the scrolling functionality.';
  String _translatedText = 'This is another block of text that might be very long and needs to be scrollable. Add more text here to see how it scrolls.';

  // Image picker instance
  final ImagePicker _picker = ImagePicker();

  // Variable to store the path of the uploaded image
  String? _uploadedImagePath;

  // Function to pick an image from the gallery, perform OCR, and update the image
  Future<void> _uploadImage() async {
    try {
      // Pick an image from the gallery
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image == null) return; // User canceled the picker

      // Perform OCR on the uploaded image
      final extractedText = await _performOCR(image.path);
      setState(() {
        _scannedText = extractedText; // Update the scanned text
        _uploadedImagePath = image.path; // Store the path of the uploaded image
      });
    } catch (e) {
      print("Error uploading image: $e");
      setState(() {
        _scannedText = 'Error uploading image.';
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
            // Display the uploaded image or the default asset image
            Positioned(
              top: 20,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  width: containerWidth, // Fixed width
                  height: containerHeight, // Fixed height
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.amber, width: 3),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(20),
                    child: _uploadedImagePath != null
                        ? Image.file(
                            File(_uploadedImagePath!),
                            width: containerWidth,
                            height: containerHeight,
                            fit: BoxFit.cover, // Ensure the image scales properly
                          )
                        : Image.asset(
                            'assets/images/upload.png',
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
                              _uploadImage(); // Trigger image upload
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
                              'UPLOAD',
                              style: TextStyle(
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