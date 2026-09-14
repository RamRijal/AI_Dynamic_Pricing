from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class PriceHistoryRecord:
    product_id: str
    old_price: float
    suggested_price: float
    final_price: float
    run_timestamp: datetime
