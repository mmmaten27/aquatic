-- Create aquaponic database
CREATE DATABASE IF NOT EXISTS aquaponic;
USE aquaponic;

-- Create sensor data table
CREATE TABLE IF NOT EXISTS sensordata (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ph DECIMAL(4,2),
    oxygen DECIMAL(4,2),
    temperature DECIMAL(4,2),
    ammonia DECIMAL(4,3),
    turbidity DECIMAL(5,2),
    conductivity DECIMAL(6,2)
);

-- Insert sample data into sensordata
INSERT INTO sensordata (ph, oxygen, temperature, ammonia, turbidity, conductivity) VALUES
(7.2, 6.5, 25.8, 0.12, 15.3, 1250.5),
(7.1, 6.8, 26.1, 0.08, 14.8, 1245.2),
(7.3, 6.2, 25.5, 0.15, 16.1, 1260.8),
(7.0, 6.9, 26.3, 0.06, 13.9, 1238.7),
(7.4, 6.1, 25.2, 0.18, 17.2, 1275.3);

-- Create alltankdata database
CREATE DATABASE IF NOT EXISTS alltankdata;
USE alltankdata;

-- Create tank1 data table
CREATE TABLE IF NOT EXISTS tank1 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ph DECIMAL(4,2),
    oxygen DECIMAL(4,2),
    temperature DECIMAL(4,2),
    ammonia DECIMAL(4,3),
    turbidity DECIMAL(5,2),
    conductivity DECIMAL(6,2)
);

-- Create tank2 data table
CREATE TABLE IF NOT EXISTS tank2 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ph DECIMAL(4,2),
    oxygen DECIMAL(4,2),
    temperature DECIMAL(4,2),
    ammonia DECIMAL(4,3),
    turbidity DECIMAL(5,2),
    conductivity DECIMAL(6,2)
);

-- Create tank3 data table
CREATE TABLE IF NOT EXISTS tank3 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ph DECIMAL(4,2),
    oxygen DECIMAL(4,2),
    temperature DECIMAL(4,2),
    ammonia DECIMAL(4,3),
    turbidity DECIMAL(5,2),
    conductivity DECIMAL(6,2)
);

-- Insert sample data into tank tables
INSERT INTO tank1 (ph, oxygen, temperature, ammonia, turbidity, conductivity) VALUES
(7.2, 6.5, 25.8, 0.12, 15.3, 1250.5);

INSERT INTO tank2 (ph, oxygen, temperature, ammonia, turbidity, conductivity) VALUES
(7.1, 6.8, 26.1, 0.08, 14.8, 1245.2);

INSERT INTO tank3 (ph, oxygen, temperature, ammonia, turbidity, conductivity) VALUES
(7.3, 6.2, 25.5, 0.15, 16.1, 1260.8);