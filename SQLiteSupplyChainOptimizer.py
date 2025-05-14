import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import sqlite3
import json
import os
import logging
from datetime import datetime, timedelta
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLiteSupplyChainOptimizer:
    def __init__(self, db_path='./supply_chain_data/supply_chain_data.db'):
        """
        Advanced Supply Chain Optimization System using SQLite
        
        :param db_path: Path to SQLite database
        """
        try:
            # SQLite Connection
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Test connection
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Logging
            logger.info(f"SQLite connection established successfully. Found tables: {[table[0] for table in tables]}")
        except Exception as e:
            logger.error(f"Failed to establish SQLite connection: {e}")
            raise
        
        # Machine Learning Models Cache
        self.ml_models = {}
    
    def _query_database(self, query, params=()):
        """Helper function to query the SQLite database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            result = []
            for row in rows:
                row_dict = dict(row)
                # Convert JSON strings back to Python objects
                for key, value in row_dict.items():
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            row_dict[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                result.append(row_dict)
            
            return result
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []
    
    def preprocess_data(self, table_name, data_type='sales'):
        """
        Advanced data preprocessing with multiple techniques
        
        :param table_name: Name of the SQLite table
        :param data_type: Type of data for specific preprocessing
        :return: Preprocessed DataFrame
        """
        try:
            # Fetch data from SQLite
            query = f"SELECT * FROM {table_name};"
            data = self._query_database(query)
            df = pd.DataFrame(data)
            
            if df.empty:
                logger.warning(f"No data found in {table_name}")
                return None
            
            # Data cleaning strategies
            if data_type == 'sales':
                # Handle sales-specific preprocessing
                df['date'] = pd.to_datetime(df['date'])
                
                # Feature engineering
                df['month'] = df['date'].dt.month
                df['quarter'] = df['date'].dt.quarter
                df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Handle missing values
                numeric_cols = ['sales_quantity', 'sales_value']
                categorical_cols = ['category', 'region', 'channel']
                
                # Imputation pipeline
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median'))
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_cols),
                        ('cat', categorical_transformer, categorical_cols)
                    ])
                
                # Normalize numerical features
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            return df
        except Exception as e:
            logger.error(f"Data preprocessing error: {e}")
            return None
    
    def demand_forecasting(self, product_id=None, forecast_horizon=90):
        """
        Demand forecasting for products
        
        :param product_id: Specific product to forecast (optional)
        :param forecast_horizon: Days to forecast ahead
        :return: Forecast results
        """
        try:
            # Get sales data
            query = "SELECT * FROM sales_data"
            if product_id:
                query += " WHERE product_id = ?"
                params = (product_id,)
            else:
                params = ()
            
            data = self._query_database(query, params)
            df = pd.DataFrame(data)
            
            if df.empty:
                logger.warning("No sales data available for forecasting")
                return None
            
            # Prepare time series data
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Group by date for overall sales forecast
            if not product_id:
                daily_sales = df.groupby('date')['sales_quantity'].sum().reset_index()
            else:
                daily_sales = df[['date', 'sales_quantity']]
            
            # Use last 30% of data as test set
            train_size = int(len(daily_sales) * 0.7)
            train_data = daily_sales[:train_size]
            test_data = daily_sales[train_size:]
            
            # Simple forecasting model using RandomForestRegressor
            # Create features from date
            for data in [train_data, test_data]:
                data['month'] = data['date'].dt.month
                data['day'] = data['date'].dt.day
                data['dayofweek'] = data['date'].dt.dayofweek
                data['quarter'] = data['date'].dt.quarter
            
            # Train the model
            features = ['month', 'day', 'dayofweek', 'quarter']
            X_train = train_data[features]
            y_train = train_data['sales_quantity']
            
            X_test = test_data[features]
            y_test = test_data['sales_quantity']
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Generate predictions for test set
            test_pred = rf.predict(X_test)
            
            # Calculate accuracy metrics
            mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
            rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
            
            # Generate dates for future forecasting
            last_date = daily_sales['date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                         periods=forecast_horizon)
            
            # Create future features
            future_df = pd.DataFrame({'date': future_dates})
            future_df['month'] = future_df['date'].dt.month
            future_df['day'] = future_df['date'].dt.day
            future_df['dayofweek'] = future_df['date'].dt.dayofweek
            future_df['quarter'] = future_df['date'].dt.quarter
            
            # Generate forecasts
            X_future = future_df[features]
            future_pred = rf.predict(X_future)
            
            future_df['forecast'] = future_pred
            
            return {
                'model_metrics': {
                    'mape': mape,
                    'rmse': rmse,
                    'feature_importance': dict(zip(features, rf.feature_importances_))
                },
                'test_predictions': {
                    'dates': test_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                    'actual': y_test.tolist(),
                    'predicted': test_pred.tolist()
                },
                'future_predictions': {
                    'dates': future_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                    'forecast': future_df['forecast'].tolist()
                }
            }
        except Exception as e:
            logger.error(f"Demand forecasting error: {e}")
            return None
    
    def supplier_analysis(self, criteria_weights=None):
        """
        Comprehensive supplier analysis with multi-criteria scoring
        
        :param criteria_weights: Custom weights for supplier evaluation criteria
        :return: Ranked and analyzed suppliers
        """
        try:
            # Get suppliers data
            query = "SELECT * FROM suppliers_data;"
            data = self._query_database(query)
            df = pd.DataFrame(data)
            
            if df.empty:
                logger.warning("No suppliers data available")
                return None
            
            # Default criteria weights if not provided
            if criteria_weights is None:
                criteria_weights = {
                    'price_rating': 0.25,
                    'quality_rating': 0.25,
                    'delivery_time': 0.20,
                    'reliability_score': 0.20,
                    'financial_stability_score': 0.10
                }
            
            # Normalize criteria
            criteria = list(criteria_weights.keys())
            df_criteria = df[criteria].copy()
            
            # For criteria where lower is better (e.g., delivery_time)
            # invert the values so higher is better
            if 'delivery_time' in criteria:
                max_delivery = df_criteria['delivery_time'].max()
                df_criteria['delivery_time'] = max_delivery - df_criteria['delivery_time']
            
            # Normalize to 0-1 scale
            scaler = MinMaxScaler()
            df_criteria[criteria] = scaler.fit_transform(df_criteria[criteria])
            
            # Calculate weighted scores
            df['total_score'] = 0
            for c, weight in criteria_weights.items():
                df['total_score'] += df_criteria[c] * weight
            
            # Rank suppliers
            df = df.sort_values('total_score', ascending=False)
            
            # Classify suppliers into risk categories
            df['risk_category'] = pd.cut(
                df['reliability_score'], 
                bins=[0, 0.3, 0.7, 1], 
                labels=['High Risk', 'Medium Risk', 'Low Risk']
            )
            
            return {
                'supplier_ranking': df[['supplier_id', 'name', 'location', 'total_score', 
                                       'price_rating', 'quality_rating', 'delivery_time', 
                                       'reliability_score', 'risk_category']].to_dict('records'),
                'risk_analysis': {
                    'high_risk_count': len(df[df['risk_category'] == 'High Risk']),
                    'medium_risk_count': len(df[df['risk_category'] == 'Medium Risk']),
                    'low_risk_count': len(df[df['risk_category'] == 'Low Risk']),
                }
            }
        except Exception as e:
            logger.error(f"Supplier analysis error: {e}")
            return None
    
    def inventory_optimization(self, product_id=None):
        """
        Inventory optimization calculations
        
        :param product_id: Specific product to optimize (optional)
        :return: Inventory optimization recommendations
        """
        try:
            # Get inventory data
            query = "SELECT * FROM inventory_data"
            if product_id:
                query += " WHERE product_id = ?"
                params = (product_id,)
            else:
                params = ()
            
            data = self._query_database(query, params)
            df = pd.DataFrame(data)
            
            if df.empty:
                logger.warning("No inventory data available")
                return None
            
            # Calculate Economic Order Quantity (EOQ) and reorder points
            results = []
            
            for _, item in df.iterrows():
                # Calculate EOQ using Wilson formula: EOQ = sqrt(2DO/H)
                # D = Annual demand, O = Order cost, H = Holding cost
                annual_demand = float(item['annual_demand'])
                order_cost = float(item['order_cost'])
                holding_cost = float(item['holding_cost'])
                
                eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
                
                # Calculate safety stock
                lead_time = float(item['lead_time'])
                lead_time_std = float(item['lead_time_std'])
                daily_demand = annual_demand / 365
                demand_std = float(item['demand_std'])
                
                # Safety stock = Z * sqrt(L * σ_d^2 + d^2 * σ_L^2)
                # Z = service level factor (1.65 for 95% service level)
                z_score = 1.65
                safety_stock = z_score * np.sqrt(
                    lead_time * demand_std**2 + daily_demand**2 * lead_time_std**2
                )
                
                # Reorder point = (Lead time * daily demand) + safety stock
                reorder_point = (lead_time * daily_demand) + safety_stock
                
                # Order frequency (days between orders)
                order_frequency = eoq / daily_demand
                
                # Total annual cost = ordering cost + holding cost
                annual_ordering_cost = (annual_demand / eoq) * order_cost
                annual_holding_cost = (eoq / 2) * holding_cost
                total_annual_cost = annual_ordering_cost + annual_holding_cost
                
                # Current vs optimal comparison
                current_eoq = float(item['economic_order_quantity'])
                current_safety = float(item['safety_stock_level'])
                current_reorder = float(item['reorder_point'])
                
                results.append({
                    'product_id': item['product_id'],
                    'category': item['category'],
                    'current_stock': float(item['current_stock']),
                    'optimal_eoq': round(eoq, 2),
                    'current_eoq': current_eoq,
                    'eoq_difference': round(eoq - current_eoq, 2),
                    'optimal_safety_stock': round(safety_stock, 2),
                    'current_safety_stock': current_safety,
                    'safety_stock_difference': round(safety_stock - current_safety, 2),
                    'optimal_reorder_point': round(reorder_point, 2),
                    'current_reorder_point': current_reorder,
                    'reorder_point_difference': round(reorder_point - current_reorder, 2),
                    'order_frequency_days': round(order_frequency, 2),
                    'total_annual_cost': round(total_annual_cost, 2),
                    'stock_status': 'Low' if float(item['current_stock']) < float(item['safety_stock_level']) else
                                   'High' if float(item['current_stock']) > 2 * float(item['reorder_point']) else 'Normal'
                })
            
            return results
        except Exception as e:
            logger.error(f"Inventory optimization error: {e}")
            return None
    
    def risk_assessment(self):
        """
        Comprehensive supply chain risk assessment
        
        :return: Risk assessment results
        """
        try:
            # Get data for risk assessment
            suppliers = pd.DataFrame(self._query_database("SELECT * FROM suppliers_data;"))
            inventory = pd.DataFrame(self._query_database("SELECT * FROM inventory_data;"))
            external = pd.DataFrame(self._query_database("SELECT * FROM external_factors_data;"))
            
            risk_assessment = {
                'supplier_risks': [],
                'inventory_risks': [],
                'external_risks': [],
                'overall_risk_score': 0
            }
            
            # Supplier risks
            if not suppliers.empty:
                for _, supplier in suppliers.iterrows():
                    risk_factors = []
                    risk_score = 0
                    
                    # Check reliability
                    if float(supplier['reliability_score']) < 0.5:
                        risk_factors.append('Low reliability score')
                        risk_score += 0.3
                    
                    # Check delivery time
                    if float(supplier['delivery_time']) > 30:
                        risk_factors.append('Long delivery time')
                        risk_score += 0.2
                    
                    # Check geopolitical risk
                    if float(supplier['geopolitical_risk_factor']) > 0.6:
                        risk_factors.append('High geopolitical risk')
                        risk_score += 0.25
                    
                    if risk_score > 0:
                        risk_assessment['supplier_risks'].append({
                            'supplier_id': supplier['supplier_id'],
                            'name': supplier['name'],
                            'risk_score': round(risk_score, 2),
                            'risk_factors': risk_factors
                        })
            
            # Inventory risks
            if not inventory.empty:
                for _, item in inventory.iterrows():
                    risk_factors = []
                    risk_score = 0
                    
                    # Check stock levels
                    if float(item['current_stock']) < float(item['safety_stock_level']):
                        risk_factors.append('Stock below safety level')
                        risk_score += 0.4
                    
                    # Check lead time variability
                    if float(item['lead_time_std']) > 5:
                        risk_factors.append('High lead time variability')
                        risk_score += 0.3
                    
                    # Check perishability for relevant items
                    if float(item['perishability']) > 0.7:
                        risk_factors.append('High perishability')
                        risk_score += 0.25
                    
                    if risk_score > 0:
                        risk_assessment['inventory_risks'].append({
                            'product_id': item['product_id'],
                            'category': item['category'],
                            'risk_score': round(risk_score, 2),
                            'risk_factors': risk_factors
                        })
            
            # External risks
            if not external.empty:
                latest_factors = external.sort_values('date', ascending=False).iloc[0]
                
                risk_factors = []
                risk_score = 0
                
                # Market sentiment
                if float(latest_factors['market_sentiment_index']) < 70:
                    risk_factors.append('Low market sentiment')
                    risk_score += 0.2
                
                # Geopolitical risk
                if float(latest_factors['geopolitical_risk_index']) > 7:
                    risk_factors.append('High geopolitical risk index')
                    risk_score += 0.3
                
                # Transportation costs
                if float(latest_factors['transportation_cost_index']) > 110:
                    risk_factors.append('Rising transportation costs')
                    risk_score += 0.25
                
                if risk_score > 0:
                    risk_assessment['external_risks'].append({
                        'date': latest_factors['date'],
                        'risk_score': round(risk_score, 2),
                        'risk_factors': risk_factors
                    })
            
            # Calculate overall risk
            supplier_risk = np.mean([r['risk_score'] for r in risk_assessment['supplier_risks']]) if risk_assessment['supplier_risks'] else 0
            inventory_risk = np.mean([r['risk_score'] for r in risk_assessment['inventory_risks']]) if risk_assessment['inventory_risks'] else 0
            external_risk = np.mean([r['risk_score'] for r in risk_assessment['external_risks']]) if risk_assessment['external_risks'] else 0
            
            risk_assessment['overall_risk_score'] = round((supplier_risk * 0.4) + (inventory_risk * 0.4) + (external_risk * 0.2), 2)
            risk_assessment['risk_level'] = 'High' if risk_assessment['overall_risk_score'] > 0.6 else 'Medium' if risk_assessment['overall_risk_score'] > 0.3 else 'Low'
            
            return risk_assessment
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return None
    
    def generate_report(self, report_type='all'):
        """
        Generate comprehensive supply chain report
        
        :param report_type: Type of report ('all', 'demand', 'supplier', 'inventory', 'risk')
        :return: Report data
        """
        report = {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'report_type': report_type
        }
        
        if report_type in ['all', 'demand']:
            report['demand_forecast'] = self.demand_forecasting()
        
        if report_type in ['all', 'supplier']:
            report['supplier_analysis'] = self.supplier_analysis()
        
        if report_type in ['all', 'inventory']:
            report['inventory_optimization'] = self.inventory_optimization()
        
        if report_type in ['all', 'risk']:
            report['risk_assessment'] = self.risk_assessment()
        
        return report
