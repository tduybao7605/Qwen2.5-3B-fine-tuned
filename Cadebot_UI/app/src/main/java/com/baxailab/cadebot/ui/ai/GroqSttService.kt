package com.baxailab.cadebot.ui.ai

import android.content.Context
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

class GroqSttService(private val context: Context) {
    private var mediaRecorder: MediaRecorder? = null
    private var audioFile: File? = null

    private val client = OkHttpClient.Builder()
        .callTimeout(30, TimeUnit.SECONDS)
        .build()

    fun startRecording() {
        audioFile = File(context.cacheDir, "cadebot_stt.m4a").also { it.delete() }
        mediaRecorder = MediaRecorder(context).apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setAudioSamplingRate(16000)
            setAudioChannels(1)
            setAudioEncodingBitRate(64000)
            setOutputFile(audioFile!!.absolutePath)
            prepare()
            start()
        }
    }

    fun stopRecording() {
        runCatching {
            mediaRecorder?.stop()
            mediaRecorder?.release()
        }
        mediaRecorder = null
    }

    suspend fun transcribe(apiKey: String): String? = withContext(Dispatchers.IO) {
        val file = audioFile?.takeIf { it.exists() && it.length() > 0 } ?: return@withContext null

        runCatching {
            val body = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file", "audio.m4a",
                    file.asRequestBody("audio/mp4".toMediaType())
                )
                .addFormDataPart("model", "whisper-large-v3-turbo")
                .addFormDataPart("language", "vi")
                .addFormDataPart("response_format", "json")
                .build()

            val request = Request.Builder()
                .url("https://api.groq.com/openai/v1/audio/transcriptions")
                .addHeader("Authorization", "Bearer $apiKey")
                .post(body)
                .build()

            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                JSONObject(response.body?.string() ?: "").optString("text")
                    .takeIf { it.isNotBlank() }
            } else null
        }.getOrNull()
    }

    fun release() {
        runCatching { mediaRecorder?.release() }
        mediaRecorder = null
    }
}
