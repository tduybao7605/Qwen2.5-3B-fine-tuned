package com.baxailab.cadebot.data.mock

import com.baxailab.cadebot.data.model.Order
import com.baxailab.cadebot.data.model.OrderStatus
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class MockOrderService @Inject constructor() {
    private val _orders = MutableStateFlow<List<Order>>(emptyList())
    val orders: StateFlow<List<Order>> = _orders

    fun placeOrder(order: Order): Order {
        val placed = order.copy(status = OrderStatus.PENDING)
        _orders.value = _orders.value + placed
        return placed
    }

    suspend fun simulatePaymentSuccess(orderId: String): Boolean {
        delay(1500)
        updateOrderStatus(orderId, OrderStatus.PAID)
        return true
    }

    fun markReady(orderId: String) = updateOrderStatus(orderId, OrderStatus.READY)

    suspend fun dispatchRobot(orderId: String): Boolean {
        updateOrderStatus(orderId, OrderStatus.DISPATCHED)
        delay(3000)
        updateOrderStatus(orderId, OrderStatus.DELIVERED)
        return true
    }

    fun confirmDelivery(orderId: String) = updateOrderStatus(orderId, OrderStatus.DELIVERED)

    private fun updateOrderStatus(orderId: String, status: OrderStatus) {
        _orders.value = _orders.value.map { if (it.orderId == orderId) it.copy(status = status) else it }
    }

    fun getOrder(orderId: String): Order? = _orders.value.find { it.orderId == orderId }
}
