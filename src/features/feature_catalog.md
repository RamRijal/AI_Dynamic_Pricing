# Feature Catalog

## Dataset Strategy

The project uses a semi-synthetic dataset design:

- a real transactional source dataset in the UCI Online Retail shape
- an enriched experiment dataset with added pricing, inventory, competitor, and behavior signals

The raw source dataset is for provenance.
The enriched experiment dataset is for training, evaluation, and pricing simulation.

## Source fields

- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

## Enriched experiment raw fields

- `product_id`
- `category`
- `date`
- `current_price`
- `units_sold`
- `stock_level`
- `competitor_price`
- `page_views`
- `unit_cost`
- `promotion_flag`
- `base_price`

## Semi-synthetic assumptions

- `product_id`: stable product key derived from source stock code
- `category`: assigned from source grouping or documented mapping
- `stock_level`: simulated from opening stock and replenishment assumptions
- `competitor_price`: simulated using a controlled band around current price
- `page_views`: simulated from demand signal plus noise
- `unit_cost`: simulated from a documented margin or cost-ratio assumption
- `promotion_flag`: simulated or inferred promotion indicator
- `base_price`: reference price used to measure discount behavior

## Engineered fields

- `price_gap`: current product price minus competitor price
- `prev_price`: previous recorded price for the same product
- `prev_units_sold`: previous observed sales for the same product
- `rolling_avg_sales`: short rolling sales context for demand trend
- `stock_ratio`: normalized stock pressure signal
- `view_to_sales_ratio`: traffic relative to realized sales
- `day_of_week`: weekday seasonality proxy
- `month`: month seasonality proxy
- `promotion_flag`: promotion indicator when available
- `discount_pct`: current discount relative to base price when available
