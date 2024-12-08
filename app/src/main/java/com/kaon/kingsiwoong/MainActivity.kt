package com.kaon.kingsiwoong

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import com.kaon.kingsiwoong.databinding.ActivityMainBinding
import java.util.*
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val TAG = MainActivity::class.java.simpleName
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var session: OrtSession
    private lateinit var activityMainBinding: ActivityMainBinding
    private val dataProcess = DataProcess(context = this)

    companion object {
        const val PERMISSION = 1
        const val PICK_IMAGE_REQUEST = 2
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        // 자동 꺼짐 해제
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // 권한 허용
        setPermissions()

        // onnx 파일 && txt 파일 불러오기
        load()

        // 갤러리에서 이미지 선택
        activityMainBinding.loadImageButton.setOnClickListener {
            openGallery()
            activityMainBinding.previewView.visibility = View.INVISIBLE
            activityMainBinding.imageView.visibility = View.VISIBLE
            activityMainBinding.rectView.visibility = View.INVISIBLE
        }

        activityMainBinding.useCameraButton.setOnClickListener {
            activityMainBinding.previewView.visibility = View.VISIBLE
            activityMainBinding.imageView.visibility = View.INVISIBLE
            activityMainBinding.rectView.visibility = View.VISIBLE
        }

        setCamera()
    }

    private fun setCamera() {
        //카메라 제공 객체
        val processCameraProvider = ProcessCameraProvider.getInstance(this).get()

        //전체 화면
        activityMainBinding.previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        // 전면 카메라
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        // 16:9 화면으로 받아옴
        val preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9).build()

        // preview 에서 받아와서 previewView 에 보여준다.
        preview.setSurfaceProvider(activityMainBinding.previewView.surfaceProvider)

        //분석 중이면 그 다음 화면이 대기중인 것이 아니라 계속 받아오는 화면으로 새로고침 함. 분석이 끝나면 그 최신 사진을 다시 분석
        val analysis = ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()

        analysis.setAnalyzer(Executors.newSingleThreadExecutor()) {
            imageProcess(it)
            it.close()
        }

        // 카메라의 수명 주기를 메인 액티비티에 귀속
        processCameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis)
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        intent.type = "image/*"
        startActivityForResult(intent, PICK_IMAGE_REQUEST)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            val selectedImageUri: Uri? = data.data
            if (selectedImageUri != null) {
                processSelectedImage(selectedImageUri)
            } else {
                Toast.makeText(this, "이미지를 선택하지 않았습니다.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun imageProcess(imageProxy: ImageProxy) {
        val bitmap = dataProcess.imageToBitmap(imageProxy)
        val floatBuilder = dataProcess.bitmapToFloatBuffer(bitmap)
        val inputName = session.inputNames.iterator().next() // session 이름

        // 모델의 요구 입력값 [1 3 640 640] [배치 사이즈, 픽셀(RGB), 너비, 높이]
        val shape = longArrayOf(
            DataProcess.BATCH_SIZE.toLong(),
            DataProcess.PIXEL_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong()
        )

        val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuilder, shape)
        val resultTensor = session.run(Collections.singletonMap(inputName, inputTensor))
        val outputs = resultTensor.get(0).value as Array<*> // [1 84 8400]
        Log.i(TAG, "outputs: ${outputs[0]}")

        val results = dataProcess.outputsToNPMSPredictions(outputs)

        Log.i(TAG, "results: $results")

        activityMainBinding.rectView.transformRect(results)
        activityMainBinding.rectView.invalidate()
    }

    private fun processSelectedImage(imageUri: Uri) {
        try {
            val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, imageUri)
            imageProcess(bitmap)
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "이미지를 처리하는 중 오류가 발생했습니다.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun resizeBitmap(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, width, height, true)
    }

    private fun imageProcess(bitmap: Bitmap) {
        val resizedBitmap = resizeBitmap(bitmap, DataProcess.INPUT_SIZE, DataProcess.INPUT_SIZE)

        val floatBuilder = dataProcess.bitmapToFloatBuffer(resizedBitmap)
        val inputName = session.inputNames.iterator().next() // session 이름

        // 모델의 요구 입력값 [1 3 640 640] [배치 사이즈, 픽셀(RGB), 너비, 높이]
        val shape = longArrayOf(
            DataProcess.BATCH_SIZE.toLong(),
            DataProcess.PIXEL_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong()
        )

        val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuilder, shape)
        val resultTensor = session.run(Collections.singletonMap(inputName, inputTensor))
        val outputs = resultTensor.get(0).value as Array<*> // [1 84 8400]
        Log.i(TAG, "outputs: ${outputs[0]}")

        val results = dataProcess.outputsToNPMSPredictions(outputs)

        Log.i(TAG, "results: $results")

        // 원본 이미지 위에 박스 그리기
        val processedBitmap = drawResultsOnBitmap(bitmap, results)

        // ImageView에 표시
        runOnUiThread {
            activityMainBinding.imageView.setImageBitmap(processedBitmap)
        }
    }

    private fun drawResultsOnBitmap(bitmap: Bitmap, results: ArrayList<Result>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = android.graphics.Canvas(mutableBitmap)
        val paint = android.graphics.Paint().apply {
            color = android.graphics.Color.RED // 박스 색상
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 2f
        }
        val textPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 25f
        }

        results.forEach { result ->
            val rectF = result.rectF
            // 박스 그리기
            canvas.drawRect(rectF, paint)
            // 클래스와 점수 표시
            canvas.drawText(
                "${dataProcess.classes[result.classIndex]} (${String.format("%.2f", result.score)})",
                rectF.left,
                rectF.top - 10,
                textPaint
            )
        }
        return mutableBitmap
    }



    private fun load() {
        dataProcess.loadModel() // onnx 모델 불러오기
        dataProcess.loadLabel() // coco txt 파일 불러오기

        ortEnvironment = OrtEnvironment.getEnvironment()
        session = ortEnvironment.createSession(
            this.filesDir.absolutePath.toString() + "/" + DataProcess.FILE_NAME,
            OrtSession.SessionOptions()
        )

        activityMainBinding.rectView.setClassLabel(dataProcess.classes)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        /*if (requestCode == PERMISSION) {
            grantResults.forEach {
                if (it != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "권한을 허용하지 않으면 사용할 수 없습니다", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
        }*/
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    private fun setPermissions() {
        val permissions = ArrayList<String>()
        permissions.add(android.Manifest.permission.READ_EXTERNAL_STORAGE)

        permissions.forEach {
            if (checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(permissions.toTypedArray(), PERMISSION)
            }
        }
    }
}
