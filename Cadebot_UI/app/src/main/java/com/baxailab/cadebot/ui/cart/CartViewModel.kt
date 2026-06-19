package com.baxailab.cadebot.ui.cart

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.baxailab.cadebot.data.mock.MockMenuService
import com.baxailab.cadebot.data.mock.MockOrderService
import com.baxailab.cadebot.data.model.CartItem
import com.baxailab.cadebot.data.model.Order
import com.baxailab.cadebot.data.model.TableInfo
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class CartUiState(
    val items: List<CartItem> = emptyList(),
    val tables: List<TableInfo> = emptyList(),
    val selectedTableId: String = "",
    val isPaymentLoading: Boolean = false,
    val paymentSuccess: Boolean = false,
    val placedOrderId: String = ""
) {
    val totalAmount: Int get() = items.sumOf { it.totalPrice }
    val isEmpty: Boolean get() = items.isEmpty()
}

@HiltViewModel
class CartViewModel @Inject constructor(
    private val orderService: MockOrderService,
    private val menuService: MockMenuService
) : ViewModel() {

    private val _uiState = MutableStateFlow(CartUiState())
    val uiState: StateFlow<CartUiState> = _uiState.asStateFlow()

    init {
        val tables = menuService.getTables()
        _uiState.value = _uiState.value.copy(
            tables = tables,
            selectedTableId = tables.firstOrNull()?.tableId ?: ""
        )
    }

    fun addItem(item: CartItem) {
        _uiState.value = _uiState.value.copy(items = _uiState.value.items + item)
    }

    fun removeItem(itemId: String) {
        _uiState.value = _uiState.value.copy(items = _uiState.value.items.filter { it.id != itemId })
    }

    fun selectTable(tableId: String) {
        _uiState.value = _uiState.value.copy(selectedTableId = tableId)
    }

    fun checkout() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isPaymentLoading = true)
            val order = Order(
                tableId = _uiState.value.selectedTableId,
                items = _uiState.value.items
            )
            val placed = orderService.placeOrder(order)
            val success = orderService.simulatePaymentSuccess(placed.orderId)
            _uiState.value = _uiState.value.copy(
                isPaymentLoading = false,
                paymentSuccess = success,
                placedOrderId = placed.orderId
            )
        }
    }

    fun clearCart() {
        _uiState.value = CartUiState(
            tables = _uiState.value.tables,
            selectedTableId = _uiState.value.selectedTableId
        )
    }
}
