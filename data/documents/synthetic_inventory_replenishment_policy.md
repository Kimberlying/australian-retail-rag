# Harbourline Retail inventory replenishment policy

**Document type:** Synthetic portfolio data — a fictional policy for the fictional retailer Harbourline Retail, written to pair with the synthetic orders and inventory dataset planned for a later version of this project.

## Reorder point

The reorder point for a product at a store is calculated as:

reorder point = (average daily unit sales × supplier lead time in days) + safety stock

Average daily unit sales use a trailing 28-day window. Promotional weeks are excluded from the average so that one-off spikes do not inflate the reorder point.

## Safety stock

Safety stock is set by product velocity class:

- **Fast movers** (top 20% of products by units sold) carry 3 days of safety stock.
- **Medium movers** carry 5 days of safety stock.
- **Slow movers** carry 7 days of safety stock, capped at 12 units per store.

Perishable products with a shelf life under 7 days carry no more than 1 day of safety stock.

## Supplier lead times

Direct-to-store suppliers deliver within 2 days. Products supplied through the distribution centre have a standard lead time of 3 days for metropolitan stores and 5 days for regional stores.

## Out-of-stock handling

A product is out of stock when available inventory is zero at the daily snapshot. Out-of-stock events on fast movers are reported to the category manager the same day. Out-of-stock rate is measured as the share of store-product combinations that are out of stock at the daily snapshot; the target is below 3%.

## Manual overrides

Store managers may raise a manual order above the system recommendation by up to 50% for a local event. Any override above 50% requires approval from the regional operations manager. The AI assistant may recommend an order quantity but must never submit a purchase order itself.

## Stocktake

Full-store stocktakes are carried out twice a year, in January and July. High-shrink categories such as razors, baby formula, and vitamins are cycle-counted weekly.
