package com.baxailab.cadebot.data.model

import java.util.UUID

data class CartItem(
    val id: String = UUID.randomUUID().toString(),
    val menuItem: MenuItem,
    val quantity: Int,
    val selectedSize: String,
    val selectedSweetness: String,
    val selectedIce: String,
    val selectedTemperature: String,
    val selectedToppings: List<String>,
    val note: String = ""
) {
    val unitPrice: Int get() {
        val toppingPrice = selectedToppings.size * 5000
        return menuItem.price + toppingPrice
    }
    val totalPrice: Int get() = unitPrice * quantity
}

data class Order(
    val orderId: String = UUID.randomUUID().toString(),
    val tableId: String,
    val items: List<CartItem>,
    val status: OrderStatus = OrderStatus.PENDING,
    val totalAmount: Int = items.sumOf { it.totalPrice },
    val createdAt: Long = System.currentTimeMillis()
)

enum class OrderStatus {
    PENDING, PAID, PREPARING, READY, DISPATCHED, DELIVERED, CANCELLED
}

data class TableInfo(
    val tableId: String,
    val tablePointId: String,
    val displayName: String,
    val zone: String
)
