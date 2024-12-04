package com.kaon.kingsiwoong

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.MappedByteBuffer

class MainActivity : AppCompatActivity() {

    private val TAG = "KingsiwoongApp"
    private lateinit var tflite: Interpreter

    private val galleryLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK && result.data != null) {
                val imageUri: Uri? = result.data?.data
                imageUri?.let {
                    try {
                        val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)
                        runModel(bitmap)
                    } catch (e: IOException) {
                        Log.e(TAG, "Error loading image from gallery: ${e.message}")
                    }
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // TFLite 모델 로드
        try {
            val tfliteModel: MappedByteBuffer = FileUtil.loadMappedFile(this, "best.tflite")
            tflite = Interpreter(tfliteModel)
        } catch (e: IOException) {
            Log.e(TAG, "Error loading TFLite model: ${e.message}")
            return
        }

        // 갤러리에서 이미지 가져오는 버튼 설정
        val loadImageButton: Button = findViewById(R.id.load_image_button)
        loadImageButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            galleryLauncher.launch(intent)
        }
    }

    // TFLite 모델로 예측 수행
    private fun runModel(bitmap: Bitmap) {
        // 모델에 입력할 Bitmap 크기를 모델의 입력 크기로 조정 (예: 640x640)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

        // 모델이 FLOAT32 데이터를 기대하는 경우, 데이터 타입을 FLOAT32로 설정하고 정규화 수행
        val inputImage = TensorImage(DataType.FLOAT32)
        inputImage.load(resizedBitmap)

        // 모델이 예상하는 입력 형상 정의 (예: 1, 640, 640, 3)
        val modelInputShape = intArrayOf(1, 640, 640, 3)

        // TensorBuffer 생성 및 데이터 로드
        val tensorBuffer = TensorBuffer.createFixedSize(modelInputShape, DataType.FLOAT32)

        // 입력 이미지를 정규화 (0 ~ 255 값을 0 ~ 1 범위로 변환)
        val floatArray = FloatArray(640 * 640 * 3)
        val bitmapPixels = IntArray(640 * 640)

        // Bitmap에서 픽셀 데이터를 얻어옴
        resizedBitmap.getPixels(bitmapPixels, 0, 640, 0, 0, 640, 640)

        // IntArray에서 floatArray로 변환 및 정규화 (0 ~ 255 -> 0.0 ~ 1.0)
        for (i in bitmapPixels.indices) {
            val pixel = bitmapPixels[i]
            floatArray[i * 3 + 0] = ((pixel shr 16) and 0xFF) / 255.0f  // Red
            floatArray[i * 3 + 1] = ((pixel shr 8) and 0xFF) / 255.0f   // Green
            floatArray[i * 3 + 2] = (pixel and 0xFF) / 255.0f           // Blue
        }

        tensorBuffer.loadArray(floatArray)

        // 모델의 출력 형상 확인 및 출력 버퍼 생성
        val outputShape = tflite.getOutputTensor(0).shape()
        Log.d(TAG, "Model output shape: ${outputShape.joinToString(", ")}")

        // 모델의 출력 형상이 예상과 다르므로, 해당 출력에 맞는 버퍼를 생성합니다.
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)

        // 추론 실행
        tflite.run(tensorBuffer.buffer, outputBuffer.buffer.rewind())

        // 모델 출력 해석 (예: (1, 75, 20, 20))
        val outputArray = outputBuffer.floatArray

        // 클래스 이름 정의
        val classes = arrayOf(
            "egg soup", "grilled mackerel", "gimbap", "doenjang stew",
            "ramen", "sea mustard soup", "cabbage kimchi", "bulgogi",
            "grilled pork belly", "stir fried zucchini", "multigrain rice", "background"
        )

        // 출력 형상 정보 추출
        val batchSize = outputShape[0]
        val channels = outputShape[1]
        val height = outputShape[2]
        val width = outputShape[3]

        // 가장 높은 확률을 가진 클래스와 그 확률을 찾음
        var maxProbability = -Float.MAX_VALUE
        var predictedClassIndex = -1

        // 모든 위치와 채널에 대해 클래스 확률을 확인
        for (h in 0 until height) {
            for (w in 0 until width) {
                for (c in 0 until classes.size) {
                    val index = c * height * width + h * width + w
                    val probability = outputArray[index]
                    if (probability > maxProbability) {
                        maxProbability = probability
                        predictedClassIndex = c
                    }
                }
            }
        }

        // 예측된 클래스가 존재할 경우 로그로 출력
        if (predictedClassIndex in classes.indices) {
            Log.d(TAG, "Predicted Class: ${classes[predictedClassIndex]}, Probability: $maxProbability")
        } else {
            Log.e(TAG, "No valid class prediction found.")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tflite.close()
    }
}