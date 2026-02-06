package com.haas.campusguard

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.*
import java.nio.FloatBuffer

class InferenceEngine(private val context: Context) {

    private var ortEnvironment: OrtEnvironment? = null
    private var yoloSession: OrtSession? = null
    private var isInitialized = false

    private var previousBoxes = mutableListOf<FloatArray>()
    private var frameCount = 0

    init {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            loadModels()
        } catch (e: Exception) {
            Log.e("InferenceEngine", "Failed to initialize: ${e.message}")
        }
    }

    private fun loadModels() {
        try {
            Log.d("InferenceEngine", "=== MODEL LOADING DEBUG ===")

            // List all files in assets
            val assetFiles = context.assets.list("") ?: emptyArray()
            Log.d("InferenceEngine", "Files in assets: ${assetFiles.joinToString()}")

            // Try to load YOLO model
            Log.d("InferenceEngine", "Loading: yolov8_person_detection.onnx")
            val modelBytes = context.assets.open("yolov8_person_detection.onnx").readBytes()

            Log.d("InferenceEngine", "Model size: ${modelBytes.size / 1024} KB")

            yoloSession = ortEnvironment?.createSession(modelBytes)
            isInitialized = true

            Log.d("InferenceEngine", "✅ MODEL LOADED!")

            // Debug: Print model info
            yoloSession?.let { session ->
                Log.d("InferenceEngine", "=== MODEL INFO ===")
                Log.d("InferenceEngine", "Input names: ${session.inputNames.joinToString()}")
                Log.d("InferenceEngine", "Output names: ${session.outputNames.joinToString()}")

                // Print input details
                session.inputInfo.forEach { (name, info) ->
                    Log.d("InferenceEngine", "Input '$name': ${info.info}")
                }

                // Print output details
                session.outputInfo.forEach { (name, info) ->
                    Log.d("InferenceEngine", "Output '$name': ${info.info}")
                }
                Log.d("InferenceEngine", "=================")
            }

        } catch (e: java.io.FileNotFoundException) {
            Log.e("InferenceEngine", "❌ FILE NOT FOUND!")
            isInitialized = false
        } catch (e: Exception) {
            Log.e("InferenceEngine", "❌ Error: ${e.message}")
            e.printStackTrace()
            isInitialized = false
        }
    }

    fun detectAnomaly(bitmap: Bitmap): DetectionResult {
        frameCount++

        // Log status every 30 frames
        if (frameCount % 30 == 0) {
            if (isInitialized) {
                Log.d("InferenceEngine", "✅ AI Mode Active - Frame $frameCount")
            } else {
                Log.w("InferenceEngine", "⚠️ Fallback Mode - Frame $frameCount (Models not loaded)")
            }
        }

        return if (isInitialized && yoloSession != null) {
            // Real AI detection
            detectWithAI(bitmap)
        } else {
            // Fallback: Rule-based detection for demo
            detectWithRules(bitmap)
        }
    }

    private fun detectWithAI(bitmap: Bitmap): DetectionResult {
        try {
            // Preprocess image for YOLO
            val inputTensor = preprocessForYOLO(bitmap)

            // Run inference
            val inputName = yoloSession?.inputNames?.firstOrNull() ?: "images"

            val inputs = mapOf(inputName to inputTensor)
            val outputs = yoloSession?.run(inputs)

            if (outputs == null || outputs.size() == 0) {
                Log.e("InferenceEngine", "No outputs from model")
                inputTensor.close()
                return DetectionResult(false, 0.0f, "No model output")
            }

            // EXTENSIVE DEBUGGING - Find out why output is null
            if (frameCount % 30 == 0) {
                Log.d("InferenceEngine", "=== OUTPUT DEBUG ===")
                Log.d("InferenceEngine", "Number of outputs: ${outputs.size()}")

                outputs.forEachIndexed { index, output ->
                    Log.d("InferenceEngine", "Output[$index] key: ${output.key}")
                    Log.d("InferenceEngine", "Output[$index] value class: ${output.value?.javaClass?.name}")
                    Log.d("InferenceEngine", "Output[$index] value: ${output.value}")
                }
            }

            // Try to get output by index
            val firstOutput = outputs.get(0)

            if (firstOutput == null) {
                Log.e("InferenceEngine", "First output (index 0) is null!")
                inputTensor.close()
                return DetectionResult(false, 0.0f, "Output index 0 null")
            }

            Log.d("InferenceEngine", "First output exists, type: ${firstOutput.javaClass.simpleName}")

            // Check if it's already an OnnxTensor
            val outputTensor: OnnxTensor = when (firstOutput) {
                is OnnxTensor -> {
                    Log.d("InferenceEngine", "✅ Output is directly OnnxTensor")
                    firstOutput
                }
                else -> {
                    // Try to get value
                    val value = firstOutput.value
                    Log.d("InferenceEngine", "Output.value type: ${value?.javaClass?.simpleName}")

                    when (value) {
                        is OnnxTensor -> {
                            Log.d("InferenceEngine", "✅ Output.value is OnnxTensor")
                            value
                        }
                        null -> {
                            Log.e("InferenceEngine", "❌ Output.value is NULL!")
                            inputTensor.close()
                            return DetectionResult(false, 0.0f, "Output.value is null")
                        }
                        else -> {
                            Log.e("InferenceEngine", "❌ Unexpected type: ${value.javaClass.name}")
                            inputTensor.close()
                            return DetectionResult(false, 0.0f, "Unexpected output type")
                        }
                    }
                }
            }

            // Now we have a valid OnnxTensor
            Log.d("InferenceEngine", "✅ Got valid OnnxTensor")

            // Get output data
            val outputBuffer = outputTensor.floatBuffer
            val outputData = FloatArray(outputBuffer.remaining())
            outputBuffer.get(outputData)

            if (frameCount % 30 == 0) {
                Log.d("InferenceEngine", "✅ AI Output size: ${outputData.size}")
            }

            // Parse YOLO output and check for anomalies
            val result = parseYOLOOutput(outputData, bitmap.width, bitmap.height)

            // Log detections
            if (result.isAnomalous) {
                Log.i("InferenceEngine", "🚨 ANOMALY: ${result.eventType} (${result.confidence})")
            } else if (frameCount % 30 == 0) {
                Log.d("InferenceEngine", "Normal: ${result.eventType} (${result.confidence})")
            }

            // Clean up
            inputTensor.close()
            outputTensor.close()

            return result

        } catch (e: Exception) {
            Log.e("InferenceEngine", "AI detection error: ${e.message}", e)
            e.printStackTrace()
            return DetectionResult(false, 0.0f, "Error: ${e.message}")
        }
    }

    private fun preprocessForYOLO(bitmap: Bitmap): OnnxTensor {
        // Resize to 640x640
        val resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

        // Convert to float array in NCHW format [1, 3, 640, 640]
        val floatArray = FloatArray(1 * 3 * 640 * 640)
        var idx = 0

        // Normalize and arrange in CHW format (channels first)
        for (c in 0 until 3) { // channels: R, G, B
            for (y in 0 until 640) {
                for (x in 0 until 640) {
                    val pixel = resized.getPixel(x, y)
                    val value = when (c) {
                        0 -> android.graphics.Color.red(pixel)
                        1 -> android.graphics.Color.green(pixel)
                        else -> android.graphics.Color.blue(pixel)
                    }
                    // Normalize to [0, 1]
                    floatArray[idx++] = value / 255.0f
                }
            }
        }

        val shape = longArrayOf(1, 3, 640, 640)
        return OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(floatArray),
            shape
        )
    }

    private fun parseYOLOOutput(outputData: FloatArray, imageWidth: Int, imageHeight: Int): DetectionResult {
        // YOLOv8 output format: [1, 84, 8400]
        // For each of 8400 anchors: [x, y, w, h, class0_conf, class1_conf, ..., class79_conf]
        // Class 0 = person in COCO dataset

        val numDetections = 8400
        val personClassIndex = 0
        val confidenceThreshold = 0.4f

        var maxPersonConfidence = 0.0f
        var bestPersonBox: FloatArray? = null
        var detectionCount = 0

        // Parse detections
        for (i in 0 until numDetections) {
            try {
                // YOLOv8 format: data is transposed, so we need to access differently
                // Each detection: index i corresponds to anchor i
                // Coordinates are at indices: i, i+8400, i+16800, i+25200
                // Class scores start at: i+33600 onwards

                val xIndex = i
                val yIndex = i + 8400
                val wIndex = i + 16800
                val hIndex = i + 25200
                val personConfIndex = i + 33600 // First class (person)

                if (personConfIndex >= outputData.size) continue

                val personConfidence = outputData[personConfIndex]

                if (personConfidence > confidenceThreshold) {
                    detectionCount++

                    if (personConfidence > maxPersonConfidence) {
                        val x = outputData[xIndex]
                        val y = outputData[yIndex]
                        val w = outputData[wIndex]
                        val h = outputData[hIndex]

                        maxPersonConfidence = personConfidence
                        bestPersonBox = floatArrayOf(x, y, w, h)
                    }
                }
            } catch (e: Exception) {
                // Skip this detection
                continue
            }
        }

        if (frameCount % 30 == 0) {
            Log.d("InferenceEngine", "Detections found: $detectionCount, Max confidence: $maxPersonConfidence")
        }

        // Analyze if we found a person
        if (bestPersonBox != null && maxPersonConfidence > confidenceThreshold) {
            val (x, y, w, h) = bestPersonBox

            // Check for anomalies based on bounding box position
            val anomaly = checkForAnomalies(x, y, w, h, 640, 640)

            return anomaly ?: DetectionResult(
                isAnomalous = false,
                confidence = maxPersonConfidence,
                eventType = "Person detected - Normal activity"
            )
        }

        // No person detected with sufficient confidence
        return DetectionResult(
            isAnomalous = false,
            confidence = maxPersonConfidence,
            eventType = "No person detected"
        )
    }

    private fun checkForAnomalies(
        x: Float,
        y: Float,
        w: Float,
        h: Float,
        imageWidth: Int,
        imageHeight: Int
    ): DetectionResult? {

        // Calculate aspect ratio
        val aspectRatio = if (h > 0) w / h else 0f

        // Calculate position in frame (normalized)
        val centerY = y / imageHeight.toFloat()
        val centerX = x / imageWidth.toFloat()

        // Rule 1: Person in bottom 25% of frame (potentially fallen)
        if (centerY > 0.75f) {
            Log.i("InferenceEngine", "⚠️ ANOMALY: Person in bottom of frame! Y: $centerY")
            return DetectionResult(
                isAnomalous = true,
                confidence = 0.88f,
                eventType = "Person Down - Emergency Possible"
            )
        }

        // Rule 2: Person very wide/flat (lying down - aspect ratio > 1.5)
        if (aspectRatio > 1.5f) {
            Log.i("InferenceEngine", "⚠️ ANOMALY: Person appears horizontal! Aspect: $aspectRatio")
            return DetectionResult(
                isAnomalous = true,
                confidence = 0.82f,
                eventType = "Person Lying Down - Unusual Position"
            )
        }

        // Rule 3: Check for rapid movement
        val currentBox = floatArrayOf(x, y, w, h)
        val displacement = calculateDisplacement(currentBox)

        if (displacement > 150f) {
            Log.i("InferenceEngine", "⚠️ ANOMALY: Rapid movement! Displacement: $displacement")
            return DetectionResult(
                isAnomalous = true,
                confidence = 0.78f,
                eventType = "Rapid Movement - Running Detected"
            )
        }

        // Update previous boxes for next frame
        if (previousBoxes.size > 5) {
            previousBoxes.removeAt(0)
        }
        previousBoxes.add(currentBox)

        return null // No anomaly
    }

    private fun calculateDisplacement(currentBox: FloatArray): Float {
        if (previousBoxes.isEmpty()) return 0f

        val prevBox = previousBoxes.last()
        val dx = currentBox[0] - prevBox[0]
        val dy = currentBox[1] - prevBox[1]

        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    private fun detectWithRules(bitmap: Bitmap): DetectionResult {
        // Fallback when AI models aren't loaded
        // Only trigger occasionally for demo purposes

        val randomValue = (0..100).random()

        return when {
            randomValue > 97 -> {
                Log.w("InferenceEngine", "⚠️ DEMO: Simulated alert (no AI loaded)")
                DetectionResult(
                    isAnomalous = true,
                    confidence = 0.65f,
                    eventType = "[DEMO MODE] Simulated Suspicious Activity"
                )
            }
            else -> DetectionResult(
                isAnomalous = false,
                confidence = 0.1f,
                eventType = "Demo mode - AI models not loaded"
            )
        }
    }

    fun cleanup() {
        yoloSession?.close()
        ortEnvironment?.close()
    }
}