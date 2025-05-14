# Supply Chain Optimization System

![Supply Chain Management](https://img.shields.io/badge/Supply%20Chain-Optimization-blue)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![SQLite](https://img.shields.io/badge/Database-SQLite-orange)
![Flask](https://img.shields.io/badge/API-Flask-lightgrey)
![Version](https://img.shields.io/badge/Version-1.0.0-success)

##  Overview

The Supply Chain Optimization System is a comprehensive data-driven solution designed to streamline and optimize supply chain operations. It combines advanced analytics, machine learning, and expert systems to provide actionable insights for inventory management, demand forecasting, supplier analysis, and risk assessment.

The system consists of three primary components:
- Core Optimization Engine (`SQLiteSupplyChainOptimizer`)
- RESTful API Backend (`backend.py`)
- Expert System for Quick Insights (`expert_system.py`)

##  Features

### Core Optimization Engine

- **Advanced Data Preprocessing**
  - Automated cleaning and normalization
  - Feature engineering for time-series data
  - Missing value imputation
  - Categorical data encoding

- **Demand Forecasting**
  - Machine learning-based time series forecasting
  - Feature importance analysis
  - Accuracy metrics (MAPE, RMSE)
  - Customizable forecast horizon

- **Supplier Analysis**
  - Multi-criteria supplier evaluation
  - Weighted scoring system
  - Risk categorization
  - Supplier ranking

- **Inventory Optimization**
  - Economic Order Quantity (EOQ) calculation
  - Safety stock determination
  - Reorder point optimization
  - Order frequency recommendations
  - Total cost analysis

- **Risk Assessment**
  - Comprehensive supply chain risk analysis
  - Supplier risk evaluation
  - Inventory risk detection
  - External factor monitoring
  - Overall risk scoring

- **Reporting**
  - Customizable report generation
  - Multiple report types

### RESTful API Backend

- **Data Access Endpoints**
  - `/api/sales` - Sales data with filtering options
  - `/api/suppliers` - Supplier data with filtering options
  - `/api/inventory` - Inventory data with filtering options
  - `/api/products` - Product master data with filtering options
  - `/api/external-factors` - External market/environmental factors

- **Aggregated Insights**
  - `/api/dashboard` - Aggregated dashboard metrics
  - `/api/expert-insights` - Expert system analysis results

- **CRUD Operations**
  - Create new records for all entities
  - Update existing records
  - Delete records

- **Cross-Origin Resource Sharing (CORS)**
  - Enabled for integration with web applications

### Expert System

- **Quick Insights**
  - Sales summary statistics
  - Inventory alerts for low stock
  - Supplier risk identification

##  Technology Stack

- **Python 3.8+** - Core programming language
- **NumPy & Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **SQLite** - Database management
- **Flask** - Web API framework
- **Flask-CORS** - Cross-Origin Resource Sharing

##  Mathematical Model
- **[Supply Chain Expert System: Supply-Chain-Optimizer](https://github.com/RiturajSingh2004/Supply-Chain-Optimizer)**

##  Data Model

The system operates on the following key data entities:

- **Sales Data** - Historical sales records
- **Suppliers Data** - Supplier information and performance metrics
- **Inventory Data** - Current stock levels and parameters
- **Product Master Data** - Product specifications and pricing
- **External Factors Data** - Market conditions and environmental factors

##  Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/supply-chain-optimizer.git
   cd supply-chain-optimizer
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure SQLite database path is correct:
   ```
   ./supply_chain_data/supply_chain_data.db
   ```

##  Usage

### Running the Expert System

```python
from expert_system import SQLiteSupplyChainExpertSystem

# Initialize the expert system
expert = SQLiteSupplyChainExpertSystem()

# Run the analysis
expert.run_expert_analysis()

# Close connection when done
expert.close()
```

### Using the Optimization Engine

```python
from SQLiteSupplyChainOptimizer import SQLiteSupplyChainOptimizer

# Initialize the optimizer
optimizer = SQLiteSupplyChainOptimizer()

# Use various optimization functions
forecast = optimizer.demand_forecasting(forecast_horizon=120)
supplier_analysis = optimizer.supplier_analysis()
inventory_recommendations = optimizer.inventory_optimization()
risks = optimizer.risk_assessment()

# Generate comprehensive report
report = optimizer.generate_report(report_type='all')
```

### Starting the API Server

```bash
python backend.py
```

The server will start on http://localhost:5000 by default.

### API Usage Examples

**Fetch Sales Data:**
```bash
curl http://localhost:5000/api/sales?limit=10&category=Electronics
```

**Add New Supplier:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"name":"New Supplier", "location":"Chicago", "price_rating":0.85, "quality_rating":0.9, "delivery_time":7, "reliability_score":0.92}' http://localhost:5000/api/suppliers
```

**Update Inventory Record:**
```bash
curl -X PUT -H "Content-Type: application/json" -d '{"product_id":"P1001", "category":"Electronics", "current_stock":150, "safety_stock_level":50, "reorder_point":75}' http://localhost:5000/api/inventory/1
```

**Delete Product Record:**
```bash
curl -X DELETE http://localhost:5000/api/products/5
```

##  Advanced Configuration

### Customizing Supplier Analysis Weights

```python
criteria_weights = {
    'price_rating': 0.35,       # Higher priority on price
    'quality_rating': 0.30,     # Strong emphasis on quality
    'delivery_time': 0.15,      # Less emphasis on delivery time
    'reliability_score': 0.15,  # Less emphasis on reliability
    'financial_stability_score': 0.05  # Minimal emphasis on financials
}

supplier_analysis = optimizer.supplier_analysis(criteria_weights=criteria_weights)
```

### Specific Product Inventory Optimization

```python
# Optimize inventory for a specific product
product_optimization = optimizer.inventory_optimization(product_id='P1001')
```

### Custom Report Generation

```python
# Generate a report focused on risk assessment
risk_report = optimizer.generate_report(report_type='risk')
```

##  Design Patterns & Architecture

The system follows several key design patterns:

1. **Repository Pattern** - Database access is abstracted through query methods
2. **Strategy Pattern** - Different optimization algorithms can be selected
3. **Factory Pattern** - Report generation based on report type
4. **Facade Pattern** - Simplified interface to complex optimization subsystems

The architecture separates concerns into:
- **Data Access Layer** - SQLite connection and queries
- **Business Logic Layer** - Optimization algorithms and analysis
- **API Layer** - RESTful endpoints for integration

##  Data Flow

1. Raw data from various sources is stored in SQLite database
2. Data is preprocessed during optimization operations
3. Machine learning models analyze patterns and generate forecasts
4. Optimization algorithms calculate ideal parameters
5. Results are made available through the API or direct method calls
6. Frontend systems can visualize and interact with the data

##  Error Handling

The system includes comprehensive error handling:

- SQLite connection errors
- Data preprocessing issues
- Machine learning model failures
- API request validation

All errors are logged with appropriate detail level for debugging.

##  Security Considerations

- Input validation on all API endpoints
- No direct SQL queries (preventing SQL injection)
- Error messages don't expose sensitive information
- CORS configuration for controlled API access

##  Performance Optimization

- Efficient SQLite queries with appropriate indices
- Caching of machine learning models in memory
- Pagination for large dataset queries
- Targeted SQL queries to minimize data transfer


##  License

This project is licensed under the MIT License - see the LICENSE file for details.


© 2025 Supply Chain Optimization Project 
