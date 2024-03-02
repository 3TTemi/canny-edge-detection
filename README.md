**Canny Edge Detection**
A basic method of image processing called edge detection is used to locate boundaries in images. Because of its accuracy, the Canny edge detection algorithm—which was first presented by John F. Canny in 1986—is a widely used technique. This program intends to perform Canny edge detection.

**Algorithms**
The algorithm accepts an image, converts it to grayscale, blurs it with a Gaussian filter, and then detects the edges. It finds the 'edge' magnitude of each pixel with a Sobel operator, then disregards all 'weak' edge pixels unless they are connected to a 'strong' edge pixel. Main steps involve: Grayscale Conversion, Noise Reduction (Gaussian Filter), Gradient Calculation and Convolution, Non-Maximum Suppression, Double threshold and Hysteresis.
