import logging
import sqlite3
import pandas as pd
import numpy as np
import os
 
class SQLiteSupplyChainExpertSystem:
    def __init__(self, db_path='./supply_chain_data/supply_chain_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)

    def fetch_table(self, table_name):
        return pd.read_sql_query(f'SELECT * FROM {table_name}', self.conn)

    def get_sales_summary(self):
        sales = self.fetch_table('sales_data')
        if sales.empty:
            return {}
        return {
            'total_sales': float(sales['sales_value'].sum()),
            'top_product': str(sales.groupby('product_id')['sales_value'].sum().idxmax()),
            'top_region': str(sales.groupby('region')['sales_value'].sum().idxmax()),
        }

    def get_inventory_alerts(self):
        inventory = self.fetch_table('inventory_data')
        if inventory.empty:
            return pd.DataFrame([])
        alerts = inventory[inventory['current_stock'] < inventory['safety_stock_level']]
        return alerts[['product_id', 'current_stock', 'safety_stock_level']]

    def get_supplier_risks(self):
        suppliers = self.fetch_table('suppliers_data')
        if suppliers.empty:
            return pd.DataFrame([])
        risky = suppliers[suppliers['reliability_score'] < 0.5]
        return risky[['supplier_id', 'name', 'reliability_score']]

    def run_expert_analysis(self):
        print("\n=== SUPPLY CHAIN EXPERT SYSTEM REPORT ===\n")
        # Demand Forecasting (simple summary)
        print("--- Sales Summary ---")
        summary = self.get_sales_summary()
        if summary:
            print(f"Total Sales Value: {summary['total_sales']:,}")
            print(f"Top Product: {summary['top_product']}")
            print(f"Top Region: {summary['top_region']}")
        else:
            print('No sales data available.')

        # Supplier Selection
        print("\n--- Supplier Risks ---")
        risky = self.get_supplier_risks()
        if not risky.empty:
            print(risky)
        else:
            print('No high-risk suppliers.')

        # Inventory Optimization (simple alert)
        print("\n--- Inventory Alerts ---")
        alerts = self.get_inventory_alerts()
        if not alerts.empty:
            print(alerts)
        else:
            print('No inventory alerts.')

        print("\n=== END OF REPORT ===\n")

    def close(self):
        self.conn.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    expert = SQLiteSupplyChainExpertSystem()
    expert.run_expert_analysis()
    expert.close()
