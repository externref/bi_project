# Supply Chain Knowledge Management System

This project provides a comprehensive Knowledge Management System for supply chain data analysis, built with Python Flask for the backend API and a modern HTML/JavaScript frontend. The data is generated synthetically to simulate a realistic supply chain environment.

## Features

- **Interactive Dashboard**: View and analyze supply chain data in an intuitive interface
- **Data Filtering**: Filter data by various criteria such as category, region, date range
- **Sorting**: Sort all tables by clicking on column headers
- **Search**: Search functionality for quickly finding specific information
- **Pagination**: Browse large datasets with efficient pagination

## Dataset Overview

The Knowledge Management System includes the following datasets:

1. **Sales Data**: Comprehensive sales information including quantity, value, region, channel, etc.
2. **Suppliers Data**: Supplier profiles with ratings, location, and performance metrics
3. **Inventory Data**: Current stock levels, safety stock, reorder points, and lead times
4. **Products Data**: Product master information with pricing, categories, and profit margins
5. **External Factors Data**: Economic indicators, market sentiment, and risk factors

## Getting Started

### Prerequisites

- Python 3.7+
- Flask
- Flask-CORS
- SQLite3

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/Supply-Chain-Optimizer.git
cd Supply-Chain-Optimizer
```

2. Install dependencies:
```
pip install flask flask-cors
```

3. Generate sample data (optional, if data is not already generated):
```
python DataGenerator.py
```

4. Start the backend server:
```
python backend.py
```

5. Open the frontend in your browser:
```
# Just open frontend.html in your web browser
```

## Usage

1. **View Data**: Open the frontend.html file in your browser to view the data
2. **Filter Data**: Use the filter dropdowns at the top of each table to narrow down your search
3. **Sort Data**: Click on any column header to sort by that column (click again to reverse sort order)
4. **Search**: Use the search boxes to find specific entries
5. **Navigate**: Use the pagination controls below each table to browse through large datasets

## API Endpoints

The backend provides the following API endpoints:

- `/api/sales` - Get sales data
- `/api/suppliers` - Get suppliers data
- `/api/inventory` - Get inventory data
- `/api/products` - Get product master data
- `/api/external-factors` - Get external factors data
- `/api/dashboard` - Get aggregated dashboard data

Each endpoint supports various query parameters for filtering data.

## Extending the Knowledge Management System

To add new visualizations or data sections:

1. Add a new tab in the HTML interface
2. Create the corresponding data fetching function in the JavaScript
3. Add any new endpoints needed in the backend.py file

## License

This project is licensed under the MIT License - see the LICENSE file for details.
