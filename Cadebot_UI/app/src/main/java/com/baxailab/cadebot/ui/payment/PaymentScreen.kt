package com.baxailab.cadebot.ui.payment

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.baxailab.cadebot.ui.components.VivaPrimaryButton
import com.baxailab.cadebot.ui.theme.*
import kotlinx.coroutines.delay

@Composable
fun PaymentScreen(
    totalAmount: Int,
    orderId: String,
    isLoading: Boolean,
    onConfirmPayment: () -> Unit,
    onSuccess: () -> Unit
) {
    val paymentSuccess = !isLoading && orderId.isNotEmpty()

    LaunchedEffect(paymentSuccess) {
        if (paymentSuccess && orderId.isNotEmpty()) {
            delay(500)
            onSuccess()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(VivaFoam),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Brush.verticalGradient(listOf(VivaEspresso, VivaCoffee)))
                .statusBarsPadding()
                .padding(vertical = 20.dp),
            contentAlignment = Alignment.Center
        ) {
            Text("Thanh toán", style = MaterialTheme.typography.headlineSmall, color = VivaOnDark)
        }

        Spacer(Modifier.height(32.dp))

        Text("Tổng thanh toán", style = MaterialTheme.typography.titleMedium, color = VivaGray)
        Spacer(Modifier.height(8.dp))
        Text(
            text = "${String.format("%,d", totalAmount)}đ",
            style = MaterialTheme.typography.displayMedium,
            color = VivaEspresso
        )

        Spacer(Modifier.height(32.dp))

        // QR placeholder
        Box(
            modifier = Modifier
                .size(220.dp)
                .clip(RoundedCornerShape(20.dp))
                .border(3.dp, VivaEspresso, RoundedCornerShape(20.dp))
                .background(VivaSurface),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("▣", style = MaterialTheme.typography.displayLarge, color = VivaEspresso)
                Spacer(Modifier.height(8.dp))
                Text("QR Demo", style = MaterialTheme.typography.titleMedium, color = VivaCoffee)
                Text("MoMo / VNPay / VietQR", style = MaterialTheme.typography.bodyMedium, color = VivaGray)
            }
        }

        Spacer(Modifier.height(16.dp))
        Text(
            text = "Quét mã QR để thanh toán\nhoặc nhấn xác nhận để giả lập",
            style = MaterialTheme.typography.bodyMedium,
            color = VivaGray,
            textAlign = TextAlign.Center
        )

        Spacer(Modifier.weight(1f))

        if (isLoading) {
            CircularProgressIndicator(color = VivaEspresso, modifier = Modifier.size(40.dp))
            Spacer(Modifier.height(12.dp))
            Text("Đang xử lý thanh toán...", style = MaterialTheme.typography.bodyMedium, color = VivaGray)
        } else {
            VivaPrimaryButton(
                text = "Xác nhận thanh toán (Demo)",
                onClick = onConfirmPayment,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp)
            )
        }
        Spacer(Modifier.height(24.dp).navigationBarsPadding())
    }
}
