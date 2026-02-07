package com.haas.campusguard

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.*
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.sqrt

@Volatile private var lastFrameForAlert: Bitmap? = null
class InferenceEngine(private val context: Context) {

    private var ortEnvironment: OrtEnvironment? = null
    private var yoloSession: OrtSession? = null
    private var isInitialized = false

    private var previousBoxes = mutableListOf<FloatArray>()
    private var frameCount = 0

    // ---- YOLOv8 assumptions (common for exported YOLOv8 detect ONNX) ----
    private val inputSize = 640
    private val numDetections = 8400
    private val numClasses = 80
    private val boxParams = 4 // x, y, w, h
    private val classesBaseOffset = numDetections * boxParams // 8400*4 = 33600

    // COCO class indices (0-based, YOLO ordering)
    // person = 0, knife = 43
    private val personClassIndex = 0
    private val knifeClassIndex = 43

    // thresholds
    private val personThreshold = 0.40f
    private val knifeThreshold = 0.35f

    // simple smoothing to reduce flicker/false positives for knife
    private var knifeHitScore = 0 // 0..3

    init {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            loadModels()
        } catch (e: Exception) {
            Log.e("InferenceEngine", "Failed to initialize: ${e.message}", e)
        }
    }

    private fun loadModels() {
        try {
            Log.d("InferenceEngine", "=== MODEL LOADING DEBUG ===")

            val assetFiles = context.assets.list("") ?: emptyArray()
            Log.d("InferenceEngine", "Files in assets: ${assetFiles.joinToString()}")

            Log.d("InferenceEngine", "Loading: yolov8_person_detection.onnx")
            val modelBytes = context.assets.open("yolov8_person_detection.onnx").readBytes()
            Log.d("InferenceEngine", "Model size: ${modelBytes.size / 1024} KB")

            yoloSession = ortEnvironment?.createSession(modelBytes)
            isInitialized = true

            Log.d("InferenceEngine", "✅ MODEL LOADED!")

            yoloSession?.let { session ->
                Log.d("InferenceEngine", "=== MODEL INFO ===")
                Log.d("InferenceEngine", "Input names: ${session.inputNames.joinToString()}")
                Log.d("InferenceEngine", "Output names: ${session.outputNames.joinToString()}")

                session.inputInfo.forEach { (name, info) ->
                    Log.d("InferenceEngine", "Input '$name': ${info.info}")
                }
                session.outputInfo.forEach { (name, info) ->
                    Log.d("InferenceEngine", "Output '$name': ${info.info}")
                }
                Log.d("InferenceEngine", "=================")
            }

        } catch (e: java.io.FileNotFoundException) {
            Log.e("InferenceEngine", "❌ FILE NOT FOUND!", e)
            isInitialized = false
        } catch (e: Exception) {
            Log.e("InferenceEngine", "❌ Error: ${e.message}", e)
            isInitialized = false
        }
    }

    fun detectAnomaly(bitmap: Bitmap): DetectionResult {
        lastFrameForAlert = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        frameCount++

        if (frameCount % 30 == 0) {
            if (isInitialized) {
                Log.d("InferenceEngine", "✅ AI Mode Active - Frame $frameCount")
            } else {
                Log.w("InferenceEngine", "⚠️ Fallback Mode - Frame $frameCount (Models not loaded)")
            }
        }

        return if (isInitialized && yoloSession != null) {
            detectWithAI(bitmap)
        } else {
            detectWithRules(bitmap)
        }
    }

    private fun detectWithAI(bitmap: Bitmap): DetectionResult {
        val session = yoloSession ?: return DetectionResult(false, 0.0f, "Session null")

        try {
            val inputTensor = preprocessForYOLO(bitmap)
            val inputName = session.inputNames.firstOrNull() ?: "images"

            val result: OrtSession.Result = session.run(mapOf(inputName to inputTensor))
            if (result.size() == 0) {
                inputTensor.close()
                result.close()
                return DetectionResult(false, 0.0f, "No model output")
            }

            val onnxValue: OnnxValue = result[0]
            val outputTensor = onnxValue as? OnnxTensor
            if (outputTensor == null) {
                inputTensor.close()
                result.close()
                return DetectionResult(false, 0.0f, "Unexpected output type: ${onnxValue.javaClass.name}")
            }

            val fb = outputTensor.floatBuffer
            val outputData = FloatArray(fb.remaining())
            fb.get(outputData)

            if (frameCount % 30 == 0) {
                Log.d("InferenceEngine", "✅ Output floats: ${outputData.size}")
            }

            // 1) Knife detection (high severity) — check first
            val knifeResult = checkKnife(outputData)
            if (knifeResult != null) {
                inputTensor.close()
                outputTensor.close()
                result.close()
                return knifeResult
            }

            // 2) Person + anomaly rules (your existing behavior)
            val personResult = parsePersonAndAnomalies(outputData)

            inputTensor.close()
            outputTensor.close()
            result.close()

            return personResult

        } catch (e: Exception) {
            Log.e("InferenceEngine", "AI detection error: ${e.message}", e)
            return DetectionResult(false, 0.0f, "Error: ${e.message}")
        }
    }

    private fun preprocessForYOLO(bitmap: Bitmap): OnnxTensor {
        val env = ortEnvironment ?: throw IllegalStateException("OrtEnvironment is null")
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        val floatArray = FloatArray(1 * 3 * inputSize * inputSize)
        var idx = 0

        // NCHW: channels first
        for (c in 0 until 3) {
            for (y in 0 until inputSize) {
                for (x in 0 until inputSize) {
                    val pixel = resized.getPixel(x, y)
                    val value = when (c) {
                        0 -> android.graphics.Color.red(pixel)
                        1 -> android.graphics.Color.green(pixel)
                        else -> android.graphics.Color.blue(pixel)
                    }
                    floatArray[idx++] = value / 255.0f
                }
            }
        }

        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArray), shape)
    }

    /**
     * Checks knife confidence across anchors.
     * Uses a simple temporal smoothing (knifeHitScore) to reduce flicker.
     */
    private fun checkKnife(outputData: FloatArray): DetectionResult? {
        // knife confidence for anchor i is:
        // output[ classesBaseOffset + (knifeClassIndex * 8400) + i ]
        val knifeBase = classesBaseOffset + (knifeClassIndex * numDetections)

        if (knifeBase >= outputData.size) {
            // Model might not be COCO-80
            if (frameCount % 60 == 0) {
                Log.w("InferenceEngine", "Knife base index out of bounds. Is this a COCO-80 model?")
            }
            knifeHitScore = max(knifeHitScore - 1, 0)
            return null
        }

        var bestKnife = 0.0f
        var bestIdx = -1

        for (i in 0 until numDetections) {
            val idx = knifeBase + i
            if (idx >= outputData.size) break
            val conf = outputData[idx]
            if (conf > bestKnife) {
                bestKnife = conf
                bestIdx = i
            }
        }

        // smoothing
        if (bestKnife > knifeThreshold) {
            knifeHitScore = minOf(knifeHitScore + 1, 3)
        } else {
            knifeHitScore = max(knifeHitScore - 1, 0)
        }

        // require it to persist a bit
        if (knifeHitScore >= 2 && bestKnife > knifeThreshold) {
            // optionally grab approximate box at same anchor (note: true knife box quality depends on model training)
            val x = outputData[bestIdx]
            val y = outputData[bestIdx + numDetections]
            val w = outputData[bestIdx + 2 * numDetections]
            val h = outputData[bestIdx + 3 * numDetections]

            Log.i("InferenceEngine", "🔪 KNIFE DETECTED conf=$bestKnife (score=$knifeHitScore)")

            return DetectionResult(
                isAnomalous = true,
                confidence = bestKnife,
                eventType = "Knife Detected - Possible Weapon Threat"
                // If your DetectionResult supports box info, this is where you'd include x,y,w,h.
            )
        }

        return null
    }

    /**
     * Keeps your existing person detection behavior:
     * - finds best person box above threshold
     * - runs your anomaly rules based on that best box
     */
    private fun parsePersonAndAnomalies(outputData: FloatArray): DetectionResult {
        val personBase = classesBaseOffset + (personClassIndex * numDetections)
        if (personBase >= outputData.size) {
            return DetectionResult(false, 0.0f, "Unexpected model output (no class scores)")
        }

        var maxPersonConfidence = 0.0f
        var bestPersonBox: FloatArray? = null
        var detectionCount = 0

        for (i in 0 until numDetections) {
            val confIdx = personBase + i
            if (confIdx >= outputData.size) break

            val personConfidence = outputData[confIdx]
            if (personConfidence > personThreshold) {
                detectionCount++
                if (personConfidence > maxPersonConfidence) {
                    val x = outputData[i]
                    val y = outputData[i + numDetections]
                    val w = outputData[i + 2 * numDetections]
                    val h = outputData[i + 3 * numDetections]
                    maxPersonConfidence = personConfidence
                    bestPersonBox = floatArrayOf(x, y, w, h)
                }
            }
        }

        if (frameCount % 30 == 0) {
            Log.d("InferenceEngine", "Persons: $detectionCount, Max person conf: $maxPersonConfidence")
        }

        if (bestPersonBox != null && maxPersonConfidence > personThreshold) {
            val (x, y, w, h) = bestPersonBox

            val anomaly = checkForAnomalies(x, y, w, h, inputSize, inputSize)
            return anomaly ?: DetectionResult(
                isAnomalous = false,
                confidence = maxPersonConfidence,
                eventType = "Person detected - Normal activity"
            )
        }

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

        val aspectRatio = if (h > 0) w / h else 0f

        val centerY = y / imageHeight.toFloat()
        // val centerX = x / imageWidth.toFloat() // (unused but you had it)

        // Rule 1: Person in bottom 25% of frame
        if (centerY > 0.75f) {
            Log.i("InferenceEngine", "⚠️ ANOMALY: Person in bottom of frame! Y: $centerY")
            return DetectionResult(
                isAnomalous = true,
                confidence = 0.88f,
                eventType = "Person Down - Emergency Possible"
            )
        }

        // Rule 2: Person very wide/flat
        if (aspectRatio > 1.5f) {
            Log.i("InferenceEngine", "⚠️ ANOMALY: Person appears horizontal! Aspect: $aspectRatio")
            return DetectionResult(
                isAnomalous = true,
                confidence = 0.82f,
                eventType = "Person Lying Down - Unusual Position"
            )
        }

        // Rule 3: rapid movement
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

        // Update previous boxes
        if (previousBoxes.size > 5) previousBoxes.removeAt(0)
        previousBoxes.add(currentBox)

        return null
    }

    private fun calculateDisplacement(currentBox: FloatArray): Float {
        if (previousBoxes.isEmpty()) return 0f
        val prevBox = previousBoxes.last()
        val dx = currentBox[0] - prevBox[0]
        val dy = currentBox[1] - prevBox[1]
        return sqrt(dx * dx + dy * dy)
    }

    private fun detectWithRules(bitmap: Bitmap): DetectionResult {
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

    fun getLastFrameForAlert(): Bitmap? = lastFrameForAlert

    fun cleanup() {
        try {
            yoloSession?.close()
        } catch (_: Exception) { }
        try {
            ortEnvironment?.close()
        } catch (_: Exception) { }
    }
}
