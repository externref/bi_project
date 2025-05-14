from flask import Flask, jsonify, request
import sqlite3
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)  # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

def query_database(query, args=()):
    """Helper function to query the SQLite database."""
    conn = sqlite3.connect('./supply_chain_data/supply_chain_data.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, args)
    rows = cursor.fetchall()
    
    # Convert JSON strings back to Python objects
    result = []
    for row in rows:
        row_dict = dict(row)
        for key, value in row_dict.items():
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                try:
                    row_dict[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        result.append(row_dict)
    
    conn.close()
    return result

@app.route('/api/sales', methods=['GET'])
def get_sales_data():
    """API endpoint to fetch sales data with optional filters."""
    limit = request.args.get('limit', 100, type=int)
    category = request.args.get('category', '')
    region = request.args.get('region', '')
    
    query = "SELECT * FROM sales_data WHERE 1=1"
    params = []
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    if region:
        query += " AND region = ?"
        params.append(region)
    
    query += f" LIMIT {limit};"
    
    data = query_database(query, params)
    return jsonify(data)

@app.route('/api/suppliers', methods=['GET'])
def get_suppliers_data():
    """API endpoint to fetch suppliers data with optional filters."""
    limit = request.args.get('limit', 100, type=int)
    location = request.args.get('location', '')
    min_rating = request.args.get('min_rating', 0, type=float)
    
    query = "SELECT * FROM suppliers_data WHERE 1=1"
    params = []
    
    if location:
        query += " AND location = ?"
        params.append(location)
    
    if min_rating > 0:
        query += " AND quality_rating >= ?"
        params.append(min_rating)
    
    query += f" LIMIT {limit};"
    
    data = query_database(query, params)
    return jsonify(data)

@app.route('/api/inventory', methods=['GET'])
def get_inventory_data():
    """API endpoint to fetch inventory data with optional filters."""
    limit = request.args.get('limit', 100, type=int)
    category = request.args.get('category', '')
    stock_status = request.args.get('stock_status', '')
    
    query = "SELECT * FROM inventory_data WHERE 1=1"
    params = []
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    # Stock status filtering is better handled on the client side
    # due to the complex conditions
    
    query += f" LIMIT {limit};"
    
    data = query_database(query, params)
    return jsonify(data)

@app.route('/api/products', methods=['GET'])
def get_product_master_data():
    """API endpoint to fetch product master data with optional filters."""
    limit = request.args.get('limit', 100, type=int)
    category = request.args.get('category', '')
    min_profit = request.args.get('min_profit', 0, type=float)
    
    query = "SELECT * FROM product_master_data WHERE 1=1"
    params = []
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    if min_profit > 0:
        query += " AND profit_margin >= ?"
        params.append(min_profit)
    
    query += f" LIMIT {limit};"
    
    data = query_database(query, params)
    return jsonify(data)

@app.route('/api/external-factors', methods=['GET'])
def get_external_factors_data():
    """API endpoint to fetch external factors data."""
    limit = request.args.get('limit', 100, type=int)
    
    query = f"SELECT * FROM external_factors_data LIMIT {limit};"
    data = query_database(query)
    return jsonify(data)

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """API endpoint to fetch aggregated data for dashboard."""
    # Get total sales
    total_sales_query = "SELECT SUM(sales_value) as total_sales FROM sales_data;"
    total_sales = query_database(total_sales_query)[0]['total_sales']
    
    # Get total inventory value
    total_inventory_query = """
    SELECT SUM(i.current_stock * p.unit_cost) as total_value 
    FROM inventory_data i 
    JOIN product_master_data p ON i.product_id = p.product_id;
    """
    try:
        total_inventory = query_database(total_inventory_query)[0]['total_value']
    except:
        total_inventory = 0  # Handle case where join might fail
    
    # Get suppliers count
    suppliers_count_query = "SELECT COUNT(*) as count FROM suppliers_data;"
    suppliers_count = query_database(suppliers_count_query)[0]['count']
    
    # Get products count
    products_count_query = "SELECT COUNT(*) as count FROM product_master_data;"
    products_count = query_database(products_count_query)[0]['count']
    
    # Get sales by category
    sales_by_category_query = """
    SELECT category, SUM(sales_value) as total 
    FROM sales_data 
    GROUP BY category 
    ORDER BY total DESC;
    """
    sales_by_category = query_database(sales_by_category_query)
    
    # Get sales by region
    sales_by_region_query = """
    SELECT region, SUM(sales_value) as total 
    FROM sales_data 
    GROUP BY region 
    ORDER BY total DESC;
    """
    sales_by_region = query_database(sales_by_region_query)
    
    return jsonify({
        'totals': {
            'sales': total_sales,
            'inventory_value': total_inventory,
            'suppliers': suppliers_count,
            'products': products_count
        },
        'sales_by_category': sales_by_category,
        'sales_by_region': sales_by_region
    })

@app.route('/api/expert-insights', methods=['GET'])
def get_expert_insights():
    """API endpoint to fetch expert system insights."""
    from expert_system import SQLiteSupplyChainExpertSystem
    expert = SQLiteSupplyChainExpertSystem()
    insights = {
        'sales_summary': expert.get_sales_summary(),
        'inventory_alerts': expert.get_inventory_alerts().to_dict(orient='records'),
        'supplier_risks': expert.get_supplier_risks().to_dict(orient='records')
    }
    expert.close()
    return jsonify(insights)

# --- CRUD API ENDPOINTS ---

def get_table_and_fields(table):
    """Helper to get table name and editable fields."""
    if table == 'sales':
        return 'sales_data', ['date', 'product_id', 'category', 'region', 'sales_channel', 'sales_quantity', 'sales_value']
    if table == 'suppliers':
        return 'suppliers_data', ['name', 'contact_person', 'location', 'price_rating', 'quality_rating', 'delivery_time', 'reliability_score', 'financial_stability_score', 'min_order_quantity', 'max_order_quantity', 'lead_time_variability', 'production_capacity', 'transportation_cost', 'compliance_certifications', 'geopolitical_risk_factor', 'environmental_sustainability_score']
    if table == 'inventory':
        return 'inventory_data', ['product_id', 'category', 'current_stock', 'safety_stock_level', 'reorder_point', 'economic_order_quantity', 'annual_demand', 'daily_average_demand', 'demand_variability', 'order_cost', 'holding_cost', 'storage_cost', 'stockout_cost', 'lead_time', 'lead_time_std', 'demand_std', 'perishability', 'shelf_life_days', 'obsolescence_risk']
    if table == 'products':
        return 'product_master_data', ['product_id', 'category', 'unit_cost', 'selling_price', 'profit_margin']
    if table == 'external-factors':
        return 'external_factors_data', ['date', 'economic_indicators', 'market_sentiment_index', 'commodity_price_index', 'geopolitical_risk_index', 'transportation_cost_index', 'weather_impact_factor', 'natural_disaster_risk']
    return None, []

@app.route('/api/<table>', methods=['POST'])
def add_record(table):
    """Add a new record to the specified table."""
    table_name, fields = get_table_and_fields(table)
    if not table_name:
        return jsonify({'error': 'Invalid table'}), 400
    data = request.json
    values = [json.dumps(data[f]) if isinstance(data.get(f), (dict, list)) else data.get(f) for f in fields]
    placeholders = ','.join(['?'] * len(fields))
    query = f"INSERT INTO {table_name} ({','.join(fields)}) VALUES ({placeholders})"
    conn = sqlite3.connect('./supply_chain_data/supply_chain_data.db')
    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/<table>/<rowid>', methods=['PUT'])
def update_record(table, rowid):
    """Update a record in the specified table by rowid."""
    table_name, fields = get_table_and_fields(table)
    if not table_name:
        return jsonify({'error': 'Invalid table'}), 400
    data = request.json
    set_clause = ', '.join([f"{f}=?" for f in fields])
    values = [json.dumps(data[f]) if isinstance(data.get(f), (dict, list)) else data.get(f) for f in fields]
    query = f"UPDATE {table_name} SET {set_clause} WHERE rowid=?"
    conn = sqlite3.connect('./supply_chain_data/supply_chain_data.db')
    cursor = conn.cursor()
    cursor.execute(query, values + [rowid])
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/<table>/<rowid>', methods=['DELETE'])
def delete_record(table, rowid):
    """Delete a record from the specified table by rowid."""
    table_name, _ = get_table_and_fields(table)
    if not table_name:
        return jsonify({'error': 'Invalid table'}), 400
    query = f"DELETE FROM {table_name} WHERE rowid=?"
    conn = sqlite3.connect('./supply_chain_data/supply_chain_data.db')
    cursor = conn.cursor()
    cursor.execute(query, [rowid])
    conn.commit()
    conn.close()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
