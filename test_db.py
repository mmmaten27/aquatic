#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database Connection Test
Testing if we can connect to aquaponic and retrieve sensor data
"""

import pymysql
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'alltankdata',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def test_connection():
    """Test basic database connection"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        print("✅ DATABASE CONNECTION: SUCCESS")
        connection.close()
        return True
    except Exception as e:
        print(f"❌ DATABASE CONNECTION FAILED: {e}")
        return False

def get_latest_sensor_data():
    """Get the most recent sensor data"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM sensordata ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
        connection.close()
        
        if result:
            print("\n✅ LATEST SENSOR DATA:")
            print(f"  ID: {result.get('id')}")
            print(f"  Timestamp: {result.get('timestamp')}")
            print(f"  pH: {result.get('ph')}")
            print(f"  Oxygen: {result.get('oxygen')} mg/L")
            print(f"  Temperature: {result.get('temperature')}°C")
            print(f"  Ammonia: {result.get('ammonia')} mg/L")
            if 'turbidity' in result:
                print(f"  Turbidity: {result.get('turbidity')}")
            if 'conductivity' in result:
                print(f"  Conductivity: {result.get('conductivity')}")
            return result
        else:
            print("❌ NO SENSOR DATA FOUND IN DATABASE")
            return None
    except Exception as e:
        print(f"❌ ERROR FETCHING SENSOR DATA: {e}")
        return None

def get_all_sensor_data():
    """Get all sensor data from database"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM sensordata")
            count_result = cursor.fetchone()
            total_records = count_result.get('count', 0)
        connection.close()
        
        if total_records > 0:
            print(f"\n✅ TOTAL RECORDS IN DATABASE: {total_records}")
            return total_records
        else:
            print("\n⚠️  DATABASE TABLE IS EMPTY")
            return 0
    except Exception as e:
        print(f"❌ ERROR COUNTING RECORDS: {e}")
        return None

def get_sensor_history(hours=24):
    """Get sensor data history for specified hours"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM sensordata
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY timestamp DESC
            """, (hours,))
            results = cursor.fetchall()
        connection.close()
        
        if results:
            print(f"\n✅ SENSOR DATA FROM LAST {hours} HOURS: {len(results)} records")
            for idx, record in enumerate(results[:5], 1):  # Show first 5 records
                print(f"  [{idx}] {record.get('timestamp')} - pH: {record.get('ph')}, Temp: {record.get('temperature')}°C")
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more records")
            return results
        else:
            print(f"\n⚠️  NO DATA FOUND FROM LAST {hours} HOURS")
            return []
    except Exception as e:
        print(f"❌ ERROR FETCHING HISTORY: {e}")
        return None

def test_api_routes():
    """Test if Flask API routes work"""
    try:
        import requests
        
        print("\n📡 TESTING API ROUTES:")
        
        # Test sensor data endpoint
        try:
            response = requests.get('http://localhost:5000/api/sensor-data', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ /api/sensor-data: OK - {data}")
            else:
                print(f"⚠️  /api/sensor-data: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ /api/sensor-data: CONNECTION FAILED - {e}")
        
        # Test history endpoint
        try:
            response = requests.get('http://localhost:5000/api/sensor-history/24', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ /api/sensor-history/24: OK - {len(data)} records")
            else:
                print(f"⚠️  /api/sensor-history/24: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ /api/sensor-history/24: CONNECTION FAILED - {e}")
            
    except ImportError:
        print("⚠️  'requests' library not installed. Skipping API tests.")
        print("   Install with: pip install requests")

def main():
    """Run all tests"""
    print("=" * 60)
    print("🔬 AQUAPONICS DATABASE CONNECTION TEST")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test 1: Connection
    if not test_connection():
        print("\n⚠️  Cannot proceed without database connection!")
        print("   Please start MySQL service first.")
        return
    
    # Test 2: Get record count
    get_all_sensor_data()
    
    # Test 3: Get latest data
    get_latest_sensor_data()
    
    # Test 4: Get history
    get_sensor_history(24)
    
    # Test 5: API routes
    test_api_routes()
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    main()
