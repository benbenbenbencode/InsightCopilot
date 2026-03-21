select
    order_id,
    user_id,
    product_id,
    order_date,
    store as city,          -- 🔥 FIX naming here
    quantity,
    price,
    quantity * price as revenue
from raw_orders